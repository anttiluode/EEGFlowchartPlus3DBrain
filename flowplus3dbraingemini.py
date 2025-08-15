# EEG Brain Source & Live Flow Chart Explorer
# Combines 3D brain visualization with real-time state-space flow diagrams

import os
import sys
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.path import Path
import matplotlib.patches as patches
import logging
import threading
import traceback
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import defaultdict

# --- Setup Logging and Suppress Warnings ---
logging.basicConfig(level=logging.WARNING)
mne.set_log_level('WARNING')
warnings.filterwarnings('ignore', message='.*QApplication.*')
warnings.filterwarnings('ignore', message='.*QWindowsWindow.*')
# Suppress specific sklearn warning about n_init
warnings.filterwarnings('ignore', category=FutureWarning, message="The default value of `n_init` will change from 10 to 'auto' in 1.4.")

# --- Define EEG Regions ---
EEG_REGIONS = {
    "Whole Brain": None,
    "Frontal": ['FP1', 'FP2', 'FZ', 'F1', 'F2', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6'],
    "Central": ['C1', 'C2', 'C3', 'C4', 'CZ', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6'],
    "Parietal": ['P1', 'P2', 'P3', 'P4', 'PZ', 'PO3', 'PO4', 'POZ'],
    "Temporal": ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8', 'TP9', 'TP10'],
    "Occipital": ['O1', 'O2', 'OZ', 'PO7', 'PO8']
}

class StateSpaceAnalysisModule:
    """
    Handles the analysis of brain state dynamics from source-reconstructed data.
    This includes feature extraction, state clustering, and transition analysis.
    """
    def __init__(self, stc_data, sfreq, n_states=8, region="Whole Brain"):
        self.stc_data = stc_data
        self.sfreq = sfreq
        self.n_states = n_states
        self.region = region
        
        # Analysis results
        self.state_labels = None
        self.transitions = defaultdict(int)
        self.epoch_times = None
        self.state_centers = None

    def analyze(self):
        """
        Runs the full state-space analysis pipeline.
        1. Extracts time-series features from epochs.
        2. Reduces dimensionality with PCA.
        3. Clusters features to define states (K-Means).
        4. Calculates transitions between states.
        """
        # 1. Extract features from short, sequential epochs
        features, self.epoch_times = self._extract_timeseries_features()
        if features.shape[0] < self.n_states:
            raise ValueError(f"Not enough epochs ({features.shape[0]}) to find {self.n_states} states. Try a longer time window.")

        # 2. Reduce dimensionality for robust clustering
        # Aim for a reasonable number of components, but not more than available epochs or features
        n_components = min(10, self.n_states * 2, features.shape[0] - 1, features.shape[1])
        if n_components < 2:
            features_reduced = features
        else:
            pca = PCA(n_components=n_components)
            features_reduced = pca.fit_transform(features)

        # 3. Cluster epochs into states
        kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
        self.state_labels = kmeans.fit_predict(features_reduced)
        self.state_centers = kmeans.cluster_centers_

        # 4. Count transitions between states
        for i in range(len(self.state_labels) - 1):
            start_state = self.state_labels[i]
            end_state = self.state_labels[i+1]
            if start_state != end_state:
                self.transitions[(start_state, end_state)] += 1

    def _extract_timeseries_features(self, epoch_duration=0.2): # 200ms epochs
        """
        Extracts features by dividing the source data into short epochs.
        The feature for each epoch is the mean power at each source vertex.
        """
        n_samples_per_epoch = int(epoch_duration * self.sfreq)
        if n_samples_per_epoch == 0: n_samples_per_epoch = 1
        
        n_vertices, n_total_samples = self.stc_data.shape
        n_epochs = n_total_samples // n_samples_per_epoch
        
        # Pre-allocate feature matrix
        features = np.zeros((n_epochs, n_vertices))
        epoch_times = np.zeros(n_epochs)
        
        for i in range(n_epochs):
            start = i * n_samples_per_epoch
            end = start + n_samples_per_epoch
            epoch_data = self.stc_data[:, start:end]
            
            # Feature: Mean power per vertex in the epoch
            power = np.mean(epoch_data ** 2, axis=1)
            features[i, :] = power
            epoch_times[i] = (start / self.sfreq) # Start time of the epoch
        
        return features, epoch_times
        
    def get_state_at_time(self, time):
        """Finds which state is active at a given time point."""
        if self.state_labels is None or self.epoch_times is None:
            return None
        # Find the index of the last epoch that started before or at the current time
        idx = np.searchsorted(self.epoch_times, time, side='right') - 1
        if 0 <= idx < len(self.state_labels):
            return self.state_labels[idx]
        return None

class EEGSourceFlowChartApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Brain Source & Live Flow Chart Explorer")
        self.root.geometry("1200x900")
        self.root.resizable(True, True)

        # --- Initialize components ---
        self.reconstructor = SourceReconstructor()
        self.preprocessing = PreprocessingPipeline()
        self.processing_thread = None
        self.brain_figures = []
        self.analysis_module = None
        self.animation_running = False
        self.stc_data, self.stc_times, self.raw_info = None, None, None
        self.current_state_text = None # For highlighting node text
        self.stop_requested = False

        self.create_ui()

    def create_ui(self):
        """Create the main user interface."""
        main_paned_window = tk.PanedWindow(self.root, orient=tk.VERTICAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)

        # --- Top frame for settings and info ---
        top_frame = tk.Frame(main_paned_window)
        main_paned_window.add(top_frame, height=350)

        # Notebook for settings tabs
        self.notebook = ttk.Notebook(top_frame)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Tabs
        self.create_standard_tab()
        self.create_flow_tab()
        self.create_advanced_tab()
        self.create_results_tab()
        
        # --- Bottom frame for visualizations and controls ---
        bottom_frame = tk.Frame(main_paned_window)
        main_paned_window.add(bottom_frame)

        viz_paned_window = tk.PanedWindow(bottom_frame, orient=tk.HORIZONTAL)
        viz_paned_window.pack(fill=tk.BOTH, expand=True)

        # Flow chart placeholder
        self.flow_chart_frame = tk.LabelFrame(viz_paned_window, text="State-Space Flow Chart ('Map of Thoughts')")
        self.flow_chart_canvas_widget = None # To hold matplotlib canvas
        viz_paned_window.add(self.flow_chart_frame, width=600)

        # Control panel
        control_panel = tk.Frame(self.root)
        control_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Status and file info
        info_frame = tk.Frame(control_panel)
        info_frame.pack(fill=tk.X, pady=5)
        self.status_label = tk.Label(info_frame, text="Ready to load EEG file...", anchor='w')
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.file_info_label = tk.Label(info_frame, text="", fg="blue", anchor='e')
        self.file_info_label.pack(side=tk.RIGHT)

        # Progress bar
        self.progress = ttk.Progressbar(control_panel, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)

        # Time controls
        self.create_time_controls(control_panel)

        # Action buttons
        self.create_action_buttons(control_panel)

    def create_time_controls(self, parent):
        time_control_frame = tk.Frame(parent)
        time_control_frame.pack(fill="x", pady=5)
        self.play_button = tk.Button(time_control_frame, text="▶ Play", command=self.toggle_playback, state=tk.DISABLED, width=8)
        self.play_button.pack(side=tk.LEFT, padx=5)
        self.time_var = tk.DoubleVar(value=0.0)
        self.time_slider = tk.Scale(time_control_frame, from_=0, to=10, orient=tk.HORIZONTAL, variable=self.time_var, command=self.on_time_change, resolution=0.01, state=tk.DISABLED)
        self.time_slider.pack(side=tk.LEFT, fill="x", expand=True, padx=10)
        self.time_label = tk.Label(time_control_frame, text="0.00s / 0.00s", width=15)
        self.time_label.pack(side=tk.LEFT)
        self.state_label = tk.Label(time_control_frame, text="Current State: --", width=15, font=("Arial", 10, "bold"))
        self.state_label.pack(side=tk.LEFT, padx=10)

    def create_action_buttons(self, parent):
        button_frame = tk.Frame(parent)
        button_frame.pack(pady=5)
        self.load_button = tk.Button(button_frame, text="Load EEG File", command=self.load_file)
        self.load_button.pack(side=tk.LEFT, padx=5)
        self.process_button = tk.Button(button_frame, text="Process & Analyze", command=self.run_analysis, state=tk.DISABLED)
        self.process_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.close_brain_button = tk.Button(button_frame, text="Close 3D Views", command=self.close_brain_views)
        self.close_brain_button.pack(side=tk.LEFT, padx=5)

    def create_standard_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Analysis Settings")
        # Frequency band, etc. (Omitted for brevity, using original code)
        freq_frame = tk.LabelFrame(frame, text="Frequency Band", font=("Arial", 10, "bold"))
        freq_frame.pack(pady=10, padx=20, fill="x")
        self.freq_var = tk.StringVar(value="alpha")
        freq_options = [("Delta (0.5-4 Hz)", "delta"), ("Theta (4-8 Hz)", "theta"), ("Alpha (8-12 Hz)", "alpha"), ("Beta (12-30 Hz)", "beta"), ("Gamma (30-50 Hz)", "gamma"), ("Broadband (0.5-50 Hz)", "broadband")]
        for i, (text, value) in enumerate(freq_options):
            tk.Radiobutton(freq_frame, text=text, variable=self.freq_var, value=value).grid(row=i//2, column=i%2, sticky="w", padx=10, pady=2)
        viz_frame = tk.LabelFrame(frame, text="3D Visualization Type", font=("Arial", 10, "bold"))
        viz_frame.pack(pady=10, padx=20, fill="x")
        self.viz_var = tk.StringVar(value="power")
        for text, value in [("Power Distribution", "power"), ("Raw Amplitude", "raw")]:
            tk.Radiobutton(viz_frame, text=text, variable=self.viz_var, value=value).pack(anchor="w", padx=10)

    def create_flow_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Flow Chart Settings")
        # Number of states
        states_frame = tk.LabelFrame(frame, text="Number of States", font=("Arial", 10, "bold"))
        states_frame.pack(pady=10, padx=20, fill="x")
        tk.Label(states_frame, text="States to detect:").pack(side=tk.LEFT, padx=5)
        self.n_states_var = tk.IntVar(value=8)
        tk.Spinbox(states_frame, from_=3, to=20, textvariable=self.n_states_var, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(states_frame, text="(More states require longer data)").pack(side=tk.LEFT, padx=10)

    def create_advanced_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Advanced Settings")
        # Preprocessing, etc. (Omitted for brevity, using original code)
        preproc_frame = tk.LabelFrame(frame, text="Preprocessing", font=("Arial", 10, "bold"))
        preproc_frame.pack(pady=10, padx=20, fill="x")
        self.remove_bad_channels_var = tk.BooleanVar(value=True)
        tk.Checkbutton(preproc_frame, text="Automatically detect and remove bad channels", variable=self.remove_bad_channels_var).pack(anchor="w", padx=10, pady=2)
        source_frame = tk.LabelFrame(frame, text="Source Reconstruction", font=("Arial", 10, "bold"))
        source_frame.pack(pady=10, padx=20, fill="x")
        tk.Label(source_frame, text="Inverse method:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.method_var = tk.StringVar(value="sLORETA")
        ttk.Combobox(source_frame, textvariable=self.method_var, values=["sLORETA", "dSPM", "MNE", "eLORETA"], state="readonly", width=15).grid(row=0, column=1, padx=10, pady=5)
        time_frame = tk.LabelFrame(frame, text="Time Window (seconds)", font=("Arial", 10, "bold"))
        time_frame.pack(pady=10, padx=20, fill="x")
        tk.Label(time_frame, text="Analyze from:").grid(row=0, column=0, padx=5, pady=5)
        self.time_start_var = tk.DoubleVar(value=0.0)
        self.time_start_spin = tk.Spinbox(time_frame, from_=0, to=1000, increment=0.5, textvariable=self.time_start_var, width=10)
        self.time_start_spin.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(time_frame, text="to").grid(row=0, column=2, padx=5, pady=5)
        self.time_end_var = tk.DoubleVar(value=10.0)
        self.time_end_spin = tk.Spinbox(time_frame, from_=0, to=1000, increment=0.5, textvariable=self.time_end_var, width=10)
        self.time_end_spin.grid(row=0, column=3, padx=5, pady=5)

    def create_results_tab(self):
        # This line is the main fix: we assign the frame to self.results_frame
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Log & Results")
        
        # Subsequent lines now correctly use self.results_frame
        self.results_text = tk.Text(self.results_frame, height=10, wrap=tk.WORD, bg="#f0f0f0")
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(self.results_text, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
    
    def on_time_change(self, value):
        if self.analysis_module is None or self.stc_data is None: return
        current_time = float(value)
        
        # Update labels
        max_time = self.time_slider['to']
        self.time_label.config(text=f"{current_time:.2f}s / {max_time:.2f}s")
        
        # Update current state display
        current_state = self.analysis_module.get_state_at_time(current_time)
        if current_state is not None:
            self.state_label.config(text=f"Current State: {current_state}")
            # Highlight node on flowchart
            if self.current_state_text and self.current_state_text.get_text() != f"State {current_state}":
                self.current_state_text.set_color('black')
                self.current_state_text.set_fontweight('normal')
            
            # Find the new text object to highlight
            ax = self.flow_chart_fig.axes[0]
            for txt in ax.texts:
                if txt.get_text() == f"State {current_state}":
                    txt.set_color('red')
                    txt.set_fontweight('bold')
                    self.current_state_text = txt
                    break
            self.flow_chart_canvas_widget.draw()
        else:
            self.state_label.config(text="Current State: --")

        # Update 3D brain view
        for brain in self.brain_figures:
            try: brain.set_time(current_time)
            except Exception: pass

    def display_flow_chart_matplotlib(self):
        """Creates and displays a Sankey-like flow chart using Matplotlib."""
        if self.flow_chart_canvas_widget:
            self.flow_chart_canvas_widget.get_tk_widget().destroy()

        self.flow_chart_fig = plt.Figure(figsize=(6, 5), dpi=100)
        ax = self.flow_chart_fig.add_subplot(111)
        ax.axis('off')

        if not self.analysis_module or not self.analysis_module.transitions:
            ax.text(0.5, 0.5, "No transitions detected.", ha='center', va='center')
        else:
            n_states = self.analysis_module.n_states
            transitions = self.analysis_module.transitions

            # --- Node positions ---
            node_positions = {}
            for i in range(n_states):
                # Simple circular layout
                angle = 2 * np.pi * i / n_states
                node_positions[i] = (np.cos(angle), np.sin(angle))
            
            # --- Draw links (curves) ---
            max_val = max(transitions.values()) if transitions else 1
            for (src, tgt), val in transitions.items():
                start_pos = node_positions[src]
                end_pos = node_positions[tgt]
                
                # Control point for bezier curve to make it arc
                mid_point = ((start_pos[0] + end_pos[0])/2, (start_pos[1] + end_pos[1])/2)
                # Push control point away from center
                ctrl_point = (mid_point[0]*0.5, mid_point[1]*0.5)

                path_data = [(Path.MOVETO, start_pos), (Path.CURVE3, ctrl_point), (Path.CURVE3, end_pos)]
                codes, verts = zip(*path_data)
                path = Path(verts, codes)
                
                linewidth = 0.5 + 3 * (val / max_val)
                patch = patches.PathPatch(path, facecolor='none', edgecolor='gray', lw=linewidth, alpha=0.6)
                ax.add_patch(patch)
            
            # --- Draw nodes ---
            for i in range(n_states):
                x, y = node_positions[i]
                ax.plot(x, y, 'o', markersize=20, color='skyblue', markeredgecolor='black', zorder=3)
                ax.text(x, y, f"State {i}", ha='center', va='center', fontsize=9, zorder=4)

            ax.set_title(f"State Transitions (Region: {self.analysis_module.region})")
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            self.flow_chart_fig.tight_layout()

        self.flow_chart_canvas_widget = FigureCanvasTkAgg(self.flow_chart_fig, master=self.flow_chart_frame)
        self.flow_chart_canvas_widget.draw()
        self.flow_chart_canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("EEG Files", "*.edf *.bdf *.fif *.set *.vhdr"), ("All files", "*.*")])
        if not filepath: return
        try:
            self.update_status("Loading EEG file...")
            self.raw = mne.io.read_raw(filepath, preload=True, verbose=False)
            self.raw.pick_types(eeg=True)
            self.filename = os.path.basename(filepath)
            
            info = f"Loaded: {self.filename} ({len(self.raw.ch_names)} ch, {self.raw.times[-1]:.1f}s, {self.raw.info['sfreq']:.0f} Hz)"
            self.file_info_label.config(text=info)
            
            # Update time spinners based on file duration
            duration = self.raw.times[-1]
            self.time_end_var.set(min(10.0, duration))
            self.time_start_spin.config(to=duration)
            self.time_end_spin.config(to=duration)
            
            self.update_status("File loaded successfully. Ready to process.")
            self.process_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            self.update_status("Failed to load file.")

    def run_analysis(self):
        """Run source reconstruction and state-space analysis in a separate thread."""
        self.stop_requested = False
        self._set_ui_state(processing=True)
        self.results_text.delete(1.0, tk.END)
        self.notebook.select(self.results_frame)
        
        self.processing_thread = threading.Thread(target=self._process_data_thread)
        self.processing_thread.start()

    def _process_data_thread(self):
        """Main processing function (runs in a separate thread)."""
        try:
            self.update_progress(10, "Step 1/7: Checking brain template...")
            subjects_dir = self._check_fsaverage()
            if self.stop_requested: return

            self.update_progress(20, "Step 2/7: Preprocessing EEG data...")
            raw_processed = self._preprocess_raw()
            self.raw_info = raw_processed.info # Save for later use
            if self.stop_requested: return
            
            self.update_progress(40, "Step 3/7: Creating forward solution...")
            fwd = self._create_forward_solution(raw_processed, subjects_dir)
            if self.stop_requested: return
            
            self.update_progress(50, "Step 4/7: Computing inverse operator...")
            inverse_operator = self._compute_inverse_operator(raw_processed, fwd)
            if self.stop_requested: return
            
            self.update_progress(60, "Step 5/7: Reconstructing sources...")
            stc = self.reconstructor.reconstruct(raw_processed, inverse_operator, self.method_var.get())
            self.stc_data, self.stc_times = stc.data, stc.times
            
            self.update_progress(75, "Step 6/7: Analyzing brain states...")
            self.analysis_module = StateSpaceAnalysisModule(
                self.stc_data, self.raw_info['sfreq'],
                n_states=self.n_states_var.get()
            )
            self.analysis_module.analyze()
            self.log_result(f"Identified {self.n_states_var.get()} states and their transitions.")

            self.update_progress(90, "Step 7/7: Preparing visualizations...")
            # For 3D plot, filter data to the selected band for visualization purposes
            raw_filtered, freq_band_name = self._filter_for_viz(raw_processed)
            stc_viz_raw = self.reconstructor.reconstruct(raw_filtered, inverse_operator, self.method_var.get())
            stc_viz, params = self._process_visualization(stc_viz_raw, freq_band_name)
            
            # --- Schedule UI updates on the main thread ---
            self.root.after(0, self._finalize_processing, stc_viz, params, subjects_dir)

        except Exception as e:
            self.update_progress(0, f"Error: {e}")
            self.log_result(f"\n✗ Error during processing: {e}\n{traceback.format_exc()}")
            self.root.after(0, lambda err=e: messagebox.showerror("Processing Error", str(err)))
        finally:
            # Ensure UI is re-enabled even if there's an error
            self.root.after(0, self._set_ui_state, False)
            
    def _finalize_processing(self, stc_viz, params, subjects_dir):
        """Updates the UI after processing is complete. Must run on main thread."""
        self.display_flow_chart_matplotlib()
        self._create_visualization(stc_viz, params, subjects_dir)
        
        max_time = self.stc_times[-1]
        self.time_slider.config(to=max_time)
        self.on_time_change("0.0")

        self.update_progress(100, "Processing complete!")
        self.log_result("\n✔ Analysis completed successfully!")

    def _preprocess_raw(self):
        """Handles the full preprocessing pipeline."""
        raw = self.raw.copy()
        raw.rename_channels({name: name.strip().replace('.', '').upper() for name in raw.ch_names}, verbose=False)
        
        # Set montage, trying several common ones
        for m_name in ['standard_1005', 'standard_1020', 'biosemi64']:
            try:
                montage = mne.channels.make_standard_montage(m_name)
                raw.set_montage(montage, match_case=False, on_missing='ignore', verbose=False)
                if any(ch['loc'][0] for ch in raw.info['chs'] if not np.isnan(ch['loc'][0])):
                    self.log_result(f"Applied {m_name} montage.")
                    break
            except Exception: continue
        else:
            self.log_result("Warning: Could not apply a standard montage. Source localization might be inaccurate.")

        if self.remove_bad_channels_var.get():
            bads = self.preprocessing.detect_bad_channels(raw)
            if bads:
                self.log_result(f"Detected bad channels: {', '.join(bads)}")
                raw.info['bads'] = bads
                raw.interpolate_bads(reset_bads=True, verbose=False)
        
        raw = self.preprocessing.remove_artifacts(raw, 'basic')
        raw.set_eeg_reference('average', projection=True, verbose=False)
        
        tmin, tmax = self.time_start_var.get(), self.time_end_var.get()
        raw.crop(tmin=tmin, tmax=tmax)
        self.log_result(f"Analyzing time window: {tmin}-{tmax} seconds")
        return raw

    def _filter_for_viz(self, raw_processed):
        """Filters data specifically for the 3D visualization."""
        raw_copy = raw_processed.copy()
        band_name = self.freq_var.get()
        if band_name != 'broadband':
            low, high = self.freq_bands[band_name]
            nyquist = raw_copy.info['sfreq'] / 2.0
            if high >= nyquist: high = nyquist - 1
            raw_copy.filter(low, high, fir_design='firwin', verbose=False)
            self.log_result(f"Filtered for visualization: {band_name} band ({low}-{high:.1f} Hz)")
        return raw_copy, band_name

    def stop_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_requested = True
            self.update_status("Stopping process...")
            self.log_result("Stop requested by user.")

    def _set_ui_state(self, processing):
        """Enable or disable UI elements based on processing state."""
        state = tk.DISABLED if processing else tk.NORMAL
        self.process_button.config(state=state)
        self.load_button.config(state=state)
        self.stop_button.config(state=tk.NORMAL if processing else tk.DISABLED)
        self.play_button.config(state=tk.NORMAL if not processing and self.stc_data is not None else tk.DISABLED)
        self.time_slider.config(state=tk.NORMAL if not processing and self.stc_data is not None else tk.DISABLED)

    # --- Toggling Playback ---
    def toggle_playback(self):
        self.animation_running = not self.animation_running
        self.play_button.config(text="⏸ Pause" if self.animation_running else "▶ Play")
        if self.animation_running: self.animate_time()

    def animate_time(self):
        if not self.animation_running or self.stc_data is None: return
        current, max_time = self.time_var.get(), self.time_slider['to']
        new_time = current + 0.05
        if new_time > max_time: new_time = 0.0 # Loop back
        self.time_var.set(new_time)
        self.on_time_change(str(new_time))
        self.root.after(50, self.animate_time)

    # --- MNE Helper/Wrapper Methods (from original code) ---
    def _create_visualization(self, stc, params, subjects_dir):
        try:
            time_label = f"{params['title']}"
            if stc.data.ndim > 1 and stc.data.shape[1] > 1: time_label += " (t=%0.2f s)"
            brain = stc.plot(subjects_dir=subjects_dir, subject='fsaverage', surface='pial', hemi='both', colormap=params['colormap'], clim=params['clim'], time_label=time_label, size=(800, 600), smoothing_steps=5, background='white', verbose=False, time_viewer=True)
            self.brain_figures.append(brain)
            self.log_result(f"Created 3D visualization with {mne.viz.get_3d_backend()}")
        except Exception as e:
            self.log_result(f"3D visualization error: {e}")
            messagebox.showwarning("3D Plot Error", f"Could not create 3D brain plot. Your system may be missing a dependency like PyVista.\n\nError: {e}")
            
    def _process_visualization(self, stc, freq_band_name):
        viz_type = self.viz_var.get()
        title = f"{viz_type.title()} - {freq_band_name.title()} Band"
        stc_meta = {'vertices': stc.vertices, 'tmin': stc.tmin, 'tstep': stc.tstep, 'subject': stc.subject}
        if viz_type == "power":
            power_data = stc.data ** 2
            stc_viz = mne.SourceEstimate(power_data, **stc_meta)
            params = {'colormap': 'hot', 'clim': dict(kind='percent', lims=[90, 95, 99]), 'title': title}
        else: # raw
            stc_viz = stc
            params = {'colormap': 'RdBu_r', 'clim': dict(kind='percent', lims=[5, 50, 95]), 'title': title}
        return stc_viz, params

    def _create_forward_solution(self, raw_processed, subjects_dir):
        self.log_result("Creating 3-layer BEM model...")
        model = mne.make_bem_model(subject='fsaverage', ico=4, conductivity=(0.3, 0.006, 0.3), subjects_dir=subjects_dir, verbose=False)
        bem_sol = mne.make_bem_solution(model, verbose=False)
        src = mne.setup_source_space('fsaverage', spacing='ico5', add_dist=False, subjects_dir=subjects_dir, verbose=False)
        fwd = mne.make_forward_solution(raw_processed.info, trans='fsaverage', src=src, bem=bem_sol, meg=False, eeg=True, mindist=5.0, verbose=False)
        return fwd
        
    def _compute_inverse_operator(self, raw_processed, fwd):
        noise_cov = mne.compute_raw_covariance(raw_processed, tmin=0, tmax=None, verbose=False)
        inverse_operator = mne.minimum_norm.make_inverse_operator(raw_processed.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=False)
        self.log_result("Created inverse operator.")
        return inverse_operator

    def _check_fsaverage(self):
        subjects_dir = os.path.join(os.path.expanduser('~'), 'mne_data')
        fsaverage_path = os.path.join(subjects_dir, 'fsaverage')
        if not os.path.isdir(fsaverage_path):
            self.log_result(f"Fsaverage template not found. Downloading to: {subjects_dir}...")
            mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)
            self.log_result("✔ Fsaverage download complete.")
        else:
            self.log_result("✔ Fsaverage brain template found.")
        return subjects_dir

    @property
    def freq_bands(self):
        return {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 50), "broadband": (0.5, 50)}
        
    def close_brain_views(self):
        for brain in self.brain_figures:
            try: brain.close()
            except Exception: pass
        self.brain_figures.clear()
        self.log_result("Closed all 3D brain views.")
        
    def log_result(self, message):
        def _log():
            self.results_text.insert(tk.END, f"{message}\n")
            self.results_text.see(tk.END)
        if self.root: self.root.after(0, _log)

    def update_status(self, message):
        def _update(): self.status_label.config(text=message)
        if self.root: self.root.after(0, _update)
        
    def update_progress(self, value, message=""):
        def _update():
            self.progress['value'] = value
            if message: self.status_label.config(text=message)
        if self.root: self.root.after(0, _update)

# --- Standalone Classes (from original code) ---
class SourceReconstructor:
    def __init__(self):
        self.methods = {'sLORETA': {'method': 'sLORETA', 'lambda2': 1.0/9.0}, 'dSPM': {'method': 'dSPM', 'lambda2': 1.0/9.0}, 'MNE': {'method': 'MNE', 'lambda2': 1.0/9.0}, 'eLORETA': {'method': 'eLORETA', 'lambda2': 1.0/9.0}}
    def reconstruct(self, raw, inverse_operator, method='sLORETA'):
        params = self.methods[method]
        return mne.minimum_norm.apply_inverse_raw(raw, inverse_operator, lambda2=params['lambda2'], method=params['method'], verbose=False)

class PreprocessingPipeline:
    @staticmethod
    def detect_bad_channels(raw, threshold=3.0):
        channel_vars = np.var(raw.get_data(), axis=1)
        median_var = np.median(channel_vars)
        mad = np.median(np.abs(channel_vars - median_var))
        if mad == 0: return []
        z_scores = 0.6745 * np.abs(channel_vars - median_var) / mad
        return [raw.ch_names[i] for i in np.where(z_scores > threshold)[0]]
    
    @staticmethod
    def remove_artifacts(raw, method='basic'):
        if method == 'basic':
            raw.filter(l_freq=0.5, h_freq=45.0, fir_design='firwin', verbose=False)
            raw.notch_filter([50, 60], fir_design='firwin', verbose=False)
        return raw

if __name__ == "__main__":
    try:
        mne.viz.set_3d_backend("pyvistaqt")
    except Exception:
        print("PyVistaQt backend not available, trying others. 3D plots might not work.")
    
    root = tk.Tk()
    app = EEGSourceFlowChartApp(root)
    root.mainloop()
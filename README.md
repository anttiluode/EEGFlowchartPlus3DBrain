# 🧠 EEG Brain Source & Live Flow Chart Explorer

![image](flowplusbrain.png)

![image2](flowplusbrain2.png)


This application provides a comprehensive toolkit for neuroscientific analysis, combining 3D brain source
localization with advanced state-space analysis of EEG data. It allows users to load EEG recordings, visualize cortical
activity in real-time, and uncover the underlying "map of thoughts" by identifying and mapping transitions between distinct
brain states.

# Features

Load and Preprocess: Handles common EEG file formats (.edf, .fif, .bdf, etc.) and performs basic preprocessing like filtering
and bad channel interpolation.

Source Reconstruction: Utilizes mne-python to perform source localization on a template brain, with multiple inverse methods available 
(sLORETA, dSPM, MNE).

Interactive 3D Visualization: Generates a 3D model of the cortical surface, displaying estimated source activity with a fully interactive
timeline and playback controls.

State-Space Analysis: Automatically identifies quasi-stable brain states from source data using machine learning (PCA and K-Means clustering).

"Map of Thoughts" Flow Chart: Displays a dynamic flow chart illustrating the discovered brain states and the probabilistic transitions between them.

Live State Highlighting: The current brain state is highlighted in real-time on the flow chart as the 3D visualization plays, linking 
temporal dynamics to abstract state representations.

Customizable Analysis: Easily adjust parameters like frequency bands, number of states, time windows, and inverse solution methods through a user-friendly GUI.

# How It Works

The application follows a sophisticated analysis pipeline to transform raw EEG signals into an intuitive map of brain dynamics:

Preprocessing: The raw EEG signal is loaded, cleaned to remove noise and artifacts, and bad channels are interpolated.

Source Reconstruction: An inverse solution (e.g., sLORETA) is applied to the clean EEG data. This process estimates the sources of the
electrical activity on a template MRI brain (fsaverage), transforming sensor-level data into brain-space data.

Feature Extraction: The resulting source data is divided into short, sequential epochs (e.g., 200ms). For each epoch, a feature vector
representing the brain's power distribution across all cortical sources is created.

State Identification: The K-Means clustering algorithm groups these high-dimensional feature vectors into a user-defined number of clusters.
Each cluster represents a recurring, quasi-stable brain "state."

Visualization: The application generates two primary, linked visualizations:

A 3D brain map showing the estimated source power over time.

A 2D flow chart built with Matplotlib, showing the discovered states as nodes and the frequency of transitions between them as weighted, 
curved edges.

# Installation

This application is built with Python and relies on several powerful scientific libraries.

Prerequisites
Python 3.8 or newer.

Setup Steps
Clone or download the repository/script.

Install the required packages:
You can install all dependencies using pip.

pip install mne scikit-learn matplotlib numpy pyvista pyvistaqt

MNE fsaverage Template:

The first time you run an analysis, the application will need to download the MNE fsaverage brain template (approx. 500MB). It will be saved
automatically to your user's home directory (e.g., C:\Users\YourUser\mne_data). This is a one-time process. Please ensure you have an internet
connection for the first run.

# Usage

Run the application from your terminal:

python flowplus3dbraingemini.py

# Load Data: Click the "Load EEG File" button to select your data file.

Configure Analysis: Adjust settings in the "Analysis Settings," "Flow Chart Settings," and "Advanced Settings" tabs as needed. For the flow chart,
a key parameter is the "Number of States" to detect.

Process & Analyze: Click the "Process & Analyze" button. The application will become unresponsive while it performs the heavy computations in
a background thread. You can monitor progress in the "Log & Results" tab.

# Explore Results: Once processing is complete, the 3D brain view and the state-space flow chart will be displayed.

Interact: Use the timeline slider and the "▶ Play" button to explore the brain dynamics over time. Observe how the highlighted state in th
e flow chart changes in sync with the activity on the 3D brain model.

# Licence MIT

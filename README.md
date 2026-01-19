#---------Processing of Magnetic Anomalies measured using AUVs----------#

# This repository contains Python script and functions for processing magnetic data
collected by the Autonomous Underwater Vehicle (AUV) like *Sentry*.

# The functions provided here enable comprehensive analysis, including spin calibration, 
coordinate rotation, filtering, temperature correction, magnetic anomaly computation, 
crossover analysis and upward continuation of anomalies  using different methods.

Features
1) Process raw magnetometer data from AUV surveys
2) Perform spin fitting calibration to correct platform-induced magnetic distortions
3) Rotate magnetometer data into local North-East-Down (NED) coordinates
4) Apply heading-dependent instrument corrections tailored to specific sensors
5) Filter data using robust smoothing techniques (including iterative reweighted least squares)
6) Calculate geomagnetic anomalies relative to the International Geomagnetic Reference Field (IGRF) model
8) Apply crossover analysis within a dive as well as across dives
7) Compute crossover correction on field anomalies
8) Export fully processed data with detailed headers to .mat and .txt files

#-------Repository Structure
	.
	├── main_processing.py              # Entry-point for magnetic data processing
	├── utils.py                        # Core processing, calibration, crossover & upcont functions
	├── Mag_2d_upcont.py                # 2-D upward continuation driver script (example)
	└── README.md                       # Project documentation                          

#-------Pre-requisites
1) Python (recommended version 3.10 or newer)

2) Python Libraries
os, re, glob, ppigrf, scipy, numpy, pandas, matplotlib, datetime, math, scikit-learn
To install them, use pip install <library name>

2) Required data files (generated or provided separately):
3) {file_prefix}_sbe49_renav.mat
   - Contains the renav_sbe49 structure with navigation data.
4) {file_prefix}_mag_renav.mat
   - Contains the renav_mag structure with magnetometer data.

Note: The {file_prefix} follows the naming convention:
sentry{SentryNumber}_{Date}_{Time}
Example: sentry744_20241204_1654

#-------Usage Instructions
Run the main driver script:
You will be prompted to enter the following:

1) Sentry number (e.g., 744)
2) Date of dive (yyyymmdd format, e.g., 20241204)
3) File start time (e.g., 1654)

The script will load the corresponding .mat files automatically.

#-------Spin Fitting Calibration:
Automatically identifies spin maneuvers where the AUV rotates on the spot.
Fits heading-dependent corrections to remove spin-induced magnetic artifacts.
Saves calibration coefficients (spin_fit_m_{prefix}.mat) and spin time windows (spin_time_m_{prefix}.mat).
Saved fits can be reused to avoid recalculation.

#-------Data Processing Options:
1) Segment: Process a short, user-selected straight segment for tuning or validation.
2) Dive (with spins): Process and save the entire dive, including spin maneuvers.
3) Dive (without spins): Process and save entire dive excluding spin maneuvers for further gridding.

Applies filtering (moving median and iterative smoothing), coordinate rotation, and instrument corrections.
Computes magnetic anomaly relative to the local IGRF model.
Outputs:
    Processed data saved to .mat and .txt files with comprehensive headers.
    Including timestamp, location, orientation, raw and corrected magnetometer data, and computed anomalies.

Function Descriptions
1) Main Workflow
	* main_processing.py
	- Entry-point driver script to run the full magnetic data processing workflow, 
	  including data loading, spin fitting, segment/dive processing, and saving.
    
1) Utils functions
    1) Mag_initialize_data
    - Initializes and structures raw magnetometer and navigation data from 
      renav files into a structured format for processing.

    1.3) Mag_process_dive_with_spins
    - Processes and saves a full dive, including spin maneuvers, 
      applying filtering, calibration, and field correction.

    1.4) Mag_process_dive_without_spins
    - Processes and saves the full dive while excluding spin sections, 
      yielding a cleaner signal for interpretation.

    1.5) Mag_process_segment
    - Processes a selected straight segment of a dive for quality control 
      and calibration verification.

    1.6) Mag_select_dive_segment
    - Identifies and selects a segment of the dive track that includes a 
      usable spin maneuver.

2) Spin Calibration and Correction
    2.1) Mag_spin_fitting
    - Detects spin maneuvers and performs calibration fitting to remove spin-related magnetic field distortions.

    2.2) Mag_fit_spin_comp
    - Fits robust sinusoidal models to magnetometer data during spins to derive spin correction coefficients.

    2.3) Mag_find_spins
    - Identifies time intervals corresponding to spin maneuvers based on navigation and altitude data.

    2.4) Mag_find_straight_segment
    - Locates straight-line track segments useful for calibration and background field analysis.

3) Sensor and Environmental Calibration
    3.1) Mag_temperature_corr
    - Applies temperature-based corrections to magnetometer readings using CALIB1520 coefficients.

    3.1) Mag_calib1520_sensors
    - Implements temperature and orientation calibration routines specific to CALIB1520 sensor standards.

    3.1) Mag_component_rotation
    - Rotates XYZ magnetometer data into the geographic North-East-Down (NED) frame using attitude (heading, pitch, roll).

4) Visualization and Plotting
    4.1) Mag_plot_filter_components
    - Applies optional filtering and visualizes the three-component magnetometer data with diagnostic plots.

5) Crossover Correction and Anomaly Adjustment
    5.1) Mag_crossover_correction
    - Performs crossover correction on magnetic anomaly data using least squares adjustment based on crossovers within or across dives and save corrected files.

Final output Data Fields Description after Crossover Correction (*_cross.txt)
------------------------------------------------------------------------------------
Field                          | Description
------------------------------------------------------------------------------------
Utime                          | Epoch time (seconds since 1970-01-01 UTC)
Lat, Lon                       | Latitude and longitude (decimal degrees)
Depth                          | Depth in kilometers
Temp                           | Temperature (°C)
Altitude                       | Vehicle altitude above seafloor (m)
Heading                        | Vehicle heading (degrees)
Roll, Pitch                    | Vehicle roll and pitch (degrees)
Mag_x, Mag_y, Mag_z            | Raw magnetometer components (nT)
Mag_N, Mag_E, Mag_D, Mag_Total | Rotated NED components and total field (nT)
Mag_*_corr                     | Heading-corrected magnetic components (nT)
Anom_*                         | Magnetic anomalies relative to IGRF (nT)
Anom_*_crossover               | Magnetic anomalies after crossover correction (nT)
------------------------------------------------------------------------------------

6) 2-D Upward Continuation of Magnetic Anomalies (GUSPI Method)
    6.1) Mag_2d_upcont.py
    - Driver script for performing 2-D upward continuation of magnetic anomaly grids.
    - Automatically estimates reasonable long (wl) and short (ws) wavelength cutoffs.

    6.2) Mag_guspi_upward
    - Function to perform 2-D upward continuation of magnetic anomaly grids using the Guspi (1987) frequency-domain method.
    - Supports variable observation depth and continuation to an arbitrary target level.
    - Applies a cosine-tapered band-pass filter to limit wavelength amplification.

7) Saving and I/O Utilities
    7.2) Mag_save_nospins
    - Saves processed dive data with spin maneuvers excluded.

    7.3) Mag_save_withspins
    - Saves processed dive data with spin maneuvers included.

    7.4) Mag_ask_yesno
    - Prompts the user for a simple yes/no response during interactive steps.

Author
Vaibhav Vijay Ingale


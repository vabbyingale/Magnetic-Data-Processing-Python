# main_processing.py
#
# Entry point for Sentry magnetic data processing
#
# Processes magnetic and SBE data from Sentry dives, performs spin calibration,
# and saves processed data in multiple forms (segment, full dive, no spins).
#
# Author: Vaibhav Vijay Ingale

import os
import warnings
from scipy.io import loadmat
import matplotlib.pyplot as plt

from utils import Mag_initialize_data, Mag_find_spins, Mag_calib1520_sensors
from utils import Mag_ask_yesno
from utils import Mag_temperature_corr, Mag_component_rotation, Mag_fit_spin_comp
from utils import Mag_plot_filter_components, Mag_spin_fitting
from utils import Mag_process_segment, Mag_process_dive_without_spins
from utils import Mag_save_nospins, Mag_process_dive_with_spins
from utils import Mag_save_withspins, Mag_select_dive_segment
from utils import Mag_crossover_correction

warnings.filterwarnings("ignore")

print("===========================================================")
print("           Sentry Magnetic Data Processing Pipeline        ")
print("===========================================================")

# -------------------------------------------------------------------------
# Step 1: Get user inputs

print("\n======================== Load the file ====================")
sentry_number = input("Enter Sentry number (e.g. 744): ")
date_part     = input("Enter date (yyyymmdd, e.g. 20241204): ")
time_part     = input("Enter file start time (e.g. 1654): ")

# Build filename prefix
file_prefix = f"sentry{sentry_number}_{date_part}_{time_part}"

# Input data files
sbe_file = f"{file_prefix}_sbe49_renav.mat"
mag_file = f"{file_prefix}_mag_renav.mat"

print(f"\nLoading sbe file: {sbe_file}")
print(f"Loading mag file: {mag_file}")

# Load MAT files
data_sbe = loadmat(sbe_file)
data_mag = loadmat(mag_file)

# Initialize structured data
magData, sbeData = Mag_initialize_data(
    data_mag["renav_mag"],
    data_sbe["renav_sbe49"]
)

del data_sbe, data_mag, sbe_file, mag_file

# -------------------------------------------------------------------------
# Step 2: Spin fitting

print("\n================ Spin fitting calibration =================")

spinFitFile  = f"spin_fit_m_{file_prefix}.mat"
spinTimeFile = f"spin_time_m_{file_prefix}.mat"

if os.path.isfile(spinFitFile):
    answ = input("Spin fit for this dive exists. Recalculate? (y/n): ")
    if answ.lower() == "y":
        print("\nRecomputing spin fit coefficients...")
        Fitting = Mag_spin_fitting(
            magData, sbeData, spinFitFile, spinTimeFile, sentry_number
        )
    else:
        print("Loading existing spin fit coefficients from file.")
        Fitting = loadmat(spinFitFile)["Fitting"]
else:
    print("Computing spin fit coefficients (first time)...")
    Fitting = Mag_spin_fitting(
        magData, sbeData, spinFitFile, spinTimeFile, sentry_number
    )

# -------------------------
# Step 3: Processing options

print("\n=================== Processing options ====================")
print("\n1: Process a short-duration segment for a quick validation.")
print("2: Process & save entire dive including spins (.mat/.txt).")
print("3: Process & save dive excluding spins at start & end.\n")

choice = input("Process? (1/2/3): ")

output_full  = f"{file_prefix}_dive_withspins.mat"
output_clean = f"{file_prefix}_dive_nospin.mat"

if choice == "1":
    print("\nChoose a segment of different duration...")
    Mag_process_segment(magData, sbeData, Fitting)

    # Show all plots and block execution until user closes them
    print("\nPlease inspect the plots. Close all plot windows to continue...")
    plt.show(block=True)
    plt.close('all')

    print("\nSegment check complete.")

    print("\nNow process and save the full dive?")
    print("2: Process & save entire dive including spins (.mat/.txt)")
    print("3: Process & save dive excluding spins at start & end")

    next_choice = input("Choose 2/3: ")

    if next_choice == "2":
        Mag_save_withspins(magData, sbeData, Fitting, file_prefix, output_full)
        plt.show(block=True)
        plt.close('all')

        print("\n----------------------------------------------------------")
        if Mag_ask_yesno("Also save part of dive excluding spins time? (y/n): "):
            Mag_save_nospins(magData, sbeData, Fitting, file_prefix, output_clean)
            plt.show(block=True)
            plt.close('all')

    elif next_choice == "3":
        Mag_save_nospins(magData, sbeData, Fitting, file_prefix, output_clean)
        plt.show(block=True)
        plt.close('all')

        print("\n----------------------------------------------------------")
        if Mag_ask_yesno("Also want to process full dive with spins? (y/n): "):
            Mag_save_withspins(magData, sbeData, Fitting, file_prefix, output_full)
            plt.show(block=True)
            plt.close('all')

    else:
        print("[WARN] Invalid follow-up choice. No processing performed.")

elif choice == "2":
    # Process & save full dive with spins
    Mag_save_withspins(magData, sbeData, Fitting, file_prefix, output_full)
    plt.show(block=True)
    plt.close('all')

    print("\n--------------------------------------------------------------------")
    if Mag_ask_yesno("Also save part of dive excluding spins time? (y/n): "):
        Mag_save_nospins(magData, sbeData, Fitting, file_prefix, output_clean)
        plt.show(block=True)
        plt.close('all')

elif choice == "3":
    # Process & save dive excluding spins
    Mag_save_nospins(magData, sbeData, Fitting, file_prefix, output_clean)
    plt.show(block=True)
    plt.close('all')

    print("\n--------------------------------------------------------------------")
    if Mag_ask_yesno("Also want to process full dive with spins? (y/n): "):
        Mag_save_withspins(magData, sbeData, Fitting, file_prefix, output_full)
        plt.show(block=True)
        plt.close('all')

else:
    print("[WARN] Invalid choice. No processing performed.")

# -------------------------------------------------------------------------
# Step 4: Crossover correction

print("\n================== Crossover Correction ===================")

if Mag_ask_yesno("Perform crossover correction? (y/n): "):
    seg_length = float(input("Enter segment length (e.g., 300): "))
    hdg_thresh = float(input("Enter heading threshold in degrees (e.g., 50): "))

    print("\nChoose crossover mode:")
    print(f"1: Perform crossover correction ONLY on this dive ({sentry_number})")
    print("2: Perform crossover correction ACROSS multiple dives")

    mode_choice = input("Select option (1/2): ")

    if mode_choice == "2":
        sentryNums = eval(input("Enter sentry number(s) as array (e.g., [744, 745]): "))
    else:
        sentryNums = [int(sentry_number)]

    for sn in sentryNums:
        print(f"\nRunning crossover correction for Sentry {sn}...")
        Mag_crossover_correction(seg_length, hdg_thresh, sn)

else:
    print("Skipped crossover + anomaly correction.")

print("\n#-------------- Magnetic data processing done! ------------#")


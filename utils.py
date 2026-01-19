# Import required libraries
import os
import re
import glob
import ppigrf
import scipy.io
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

# Import required modules
from pathlib import Path
from numpy.fft import fft
from scipy.io import loadmat
from scipy.io import savemat
from scipy.signal import resample
from scipy.signal import convolve
from scipy.special import factorial
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from datetime import datetime, timezone
from numpy.fft import fft2, ifft2, ifftshift
from scipy.signal import medfilt, savgol_filter
from math import radians, sin, cos, sqrt, atan2
from sklearn.linear_model import LinearRegression

def Mag_initialize_data(renav_mag, renav_sbe49):
    """
    Initializes structured data from raw renav inputs
    MATLAB equivalent of Mag_initialize_data.m

    Parameters
    ----------
    renav_mag : dict
        Loaded MATLAB struct renav_mag (from loadmat)
    renav_sbe49 : dict
        Loaded MATLAB struct renav_sbe49 (from loadmat)

    Returns
    -------
    mag : dict
        Magnetometer structured data
    sbe : dict
        SBE49 structured data
    """

    print("\nTo verify variables, refer to function Mag_initialize_data")

    # ---------------------------------------------------------------------
    # Step 1: SBE data
    sbe_data = renav_sbe49[0, 0] if isinstance(renav_sbe49, np.ndarray) else renav_sbe49

    sbe = {}
    sbe['utime'] = sbe_data['t'].squeeze()
    sbe['dtime'] = [datetime.fromtimestamp(float(t), tz=timezone.utc) for t in sbe['utime']]
    sbe['lat'] = sbe_data['renavLat'].squeeze()
    sbe['lon'] = sbe_data['renavLon'].squeeze()
    sbe['depth'] = sbe_data['renavDepth'].squeeze()
    sbe['heading'] = sbe_data['renavHeading'].squeeze()
    sbe['alt'] = sbe_data['renavAltitude'].squeeze()
    sbe['temp'] = sbe_data['temperature'].squeeze()
    sbe['samp_rate'] = round(1 / ((sbe['utime'][-1] - sbe['utime'][0]) / len(sbe['utime'])))

    # ---------------------------------------------------------------------
    # Step 2: MAG data
    mag_data = renav_mag[0, 0] if isinstance(renav_mag, np.ndarray) else renav_mag
    maggie_top = mag_data['maggie_top'][0, 0]  
    
    mag = {}
    mag['utime'] = maggie_top['t'].squeeze()
    mag['dtime'] = [datetime.fromtimestamp(float(t), tz=timezone.utc) for t in mag['utime']]
    mag['x'] = maggie_top['x'].squeeze()
    mag['y'] = maggie_top['y'].squeeze()
    mag['z'] = maggie_top['z'].squeeze()
    mag['temp'] = maggie_top['temperature'].squeeze()
    mag['heading'] = maggie_top['renavHeading'].squeeze()
    mag['roll'] = maggie_top['renavRoll'].squeeze()
    mag['pitch'] = maggie_top['renavPitch'].squeeze()
    mag['lat'] = maggie_top['renavLat'].squeeze()
    mag['lon'] = maggie_top['renavLon'].squeeze()
    mag['samp_rate'] = round(1 / ((mag['utime'][-1] - mag['utime'][0]) / len(mag['utime'])))

    return mag, sbe

def Mag_ask_yesno(prompt):
    """
    Prompt the user for a yes/no response.

    Parameters
    ----------
    prompt : str
        The message to display to the user.

    Returns
    -------
    flag : bool
        True if user inputs 'y' or 'Y', False otherwise.
    """
    reply = input(prompt + " [y/n]: ")
    return reply.strip().lower() == 'y'

def Mag_find_spins(sbe_heading, sbe_lon, sbe_lat, sbe_utime, sbe_alt):
    """
    Identify and select a time segment corresponding to only spin maneuver.

    Returns:
        spin_start: Unix time marking start of selected spin
        spin_end: Unix time marking end of selected spin
    """
    # ---------------------------------------------------------------------
    # Step 1: Identify heading jumps
    ang = np.diff(sbe_heading)
    ang = np.append(ang, 0)  # pad to match original length
    ndc = np.where(np.abs(ang) > 180)[0]  # heading jumps >180 deg

	# ---------------------------------------------------------------------
    # Step 2: Plotting
    fig1, ax1 = plt.subplots()
    ax1.plot(sbe_lon, sbe_lat, 'b-', linewidth=0.8, label='Track')
    ax1.plot(sbe_lon[ndc], sbe_lat[ndc], 'ro', label='Heading change')
    ax1.set_title('Track with Heading Jumps')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.plot(sbe_utime, sbe_alt, 'b-', linewidth=0.8)
    ax2.plot(sbe_utime[ndc], sbe_alt[ndc], 'ro')
    ax2.set_title('Altitude vs Time')
    ax2.set_xlabel('Unix Time')
    ax2.set_ylabel('Altitude (m)')

    fig3, ax3 = plt.subplots()
    ax3.plot(sbe_utime, sbe_heading, 'b-', linewidth=0.8)
    ax3.plot(sbe_utime[ndc], sbe_heading[ndc], 'ro')
    ax3.set_title('Heading vs Time')
    ax3.set_xlabel('Unix Time')
    ax3.set_ylabel('Heading (deg)')

    plt.tight_layout()
    plt.ion()
    plt.show(block=False)

    print("\nZoom/pan in Figure 3 to the spin region, then press any key to start point selection...")

    # Storage for return values from inside event
    nonlocal_results = []

    def on_key(event):
        print("Key pressed. Now click two points on Figure 3...")
        fig3.canvas.mpl_disconnect(cid_key)

        clicked_pts = plt.ginput(2, timeout=-1)
        p1, p2 = clicked_pts[0][0], clicked_pts[1][0]

        spin_start = min(p1, p2)
        spin_end = max(p1, p2)

        print(f"Selected spin limits: start={spin_start:.2f}, end={spin_end:.2f}")

        # Zoom in on selected range in the figures
        ax3.set_xlim(spin_start, spin_end)
        ax2.set_xlim(spin_start, spin_end)

        # Plot the spins in new figure
        fig4, ax4 = plt.subplots()
        track_ndc = np.where((sbe_utime > spin_start) & (sbe_utime < spin_end))[0]
        ax4.plot(sbe_lon[track_ndc], sbe_lat[track_ndc], 'b-', linewidth=0.8)
        ax4.set_title('Spin Maneuver Track Segment')
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        plt.show()

        # Store results
        nonlocal_results.append((spin_start, spin_end))

        # --- Close all figures automatically after selection ---
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        # Optional: keep fig4 open for inspection
        # plt.close(fig4)

    # Connect key press event to Figure 3
    cid_key = fig3.canvas.mpl_connect('key_press_event', on_key)

    plt.show(block=True)  # Wait until key press and clicks

    # Return the spin limits if selected
    if nonlocal_results:
        return nonlocal_results[0]
    else:
        return None, None

def Mag_calib1520_sensors(X, sensor):
    """
    Applies temperature and orientation calibration to 3-axis magnetometer data.

    Parameters
    ----------
    X : ndarray, shape (N, 4)
        Raw measurements [X Y Z Temp], in gauss and °C
    sensor : int
        Sensor serial number (e.g., 687, 688, 689, 690) to select calibration curves.

    Returns
    -------
    COR : ndarray, shape (N, 3)
        Corrected field components (nT)
    fcorT : ndarray, shape (N,)
        Total corrected field strength (nT)
    """

    # ---------------------------------------------------------------------
    # Step 1: Convert input gauss measurements to nT
    raw = np.zeros((X.shape[0], 3))
    raw[:, 0] = X[:, 0] * 1e5  # X component
    raw[:, 1] = X[:, 1] * 1e5  # Y component
    raw[:, 2] = X[:, 2] * 1e5  # Z component
    T = X[:, 3]                # Temperature

    # ---------------------------------------------------------------------
    # Step 2: Select sensor-specific calibration coefficients
    if sensor == 687:
        print("Using sensor 0687")
        b = np.column_stack([
            25.837 + 1.3642 * T - 0.058418 * T**2,
            84.498 - 4.6222 * T + 0.048285 * T**2,
            37.862 - 2.8216 * T + 0.0032608 * T**2
        ])
        scale = np.column_stack([
            0.99397 + 2.1563e-4 * T + 4.1443e-7 * T**2,
            0.99141 + 2.6452e-4 * T + 5.8845e-6 * T**2,
            0.99623 + 1.7733e-4 * T + 1.6337e-6 * T**2
        ])
        r = np.column_stack([
            -197.81 + 33.499 * T - 0.69658 * T**2,
            375.91 + 17.633 * T - 0.46883 * T**2,
            238.68 - 11.479 * T + 0.044622 * T**2
        ])

    elif sensor == 688:
        print("Using sensor 0688")
        b = np.column_stack([
            108.28 - 4.4685 * T + 1.5674e-3 * T**2,
            31.393 - 0.72391 * T - 3.0317e-2 * T**2,
            -45.188 + 1.6138 * T - 9.6169e-3 * T**2
        ])
        scale = np.column_stack([
            0.99784 + 6.976e-6 * T + 6.4954e-6 * T**2,
            0.98097 + 4.6894e-4 * T - 5.682e-6 * T**2,
            0.99800 + 2.9658e-5 * T + 5.8352e-6 * T**2
        ])
        r = np.column_stack([
            -199.24 + 11.182 * T - 0.42256 * T**2,
            282.09 - 8.9838 * T - 0.084534 * T**2,
            42.962 - 20.321 * T + 0.28916 * T**2
        ])

    elif sensor == 689:
        print("Using sensor 0689")
        b = np.column_stack([
            32.603 - 1.7135 * T + 6.7599e-2 * T**2,
            -62.389 + 7.4899 * T - 1.5924e-1 * T**2,
            42.883 - 1.2902 * T - 5.5892e-2 * T**2
        ])
        scale = np.column_stack([
            0.99537 + 1.142e-4 * T + 7.7696e-6 * T**2,
            0.99078 + 2.468e-4 * T + 7.1949e-6 * T**2,
            0.99659 + 1.331e-5 * T + 8.2567e-6 * T**2
        ])
        r = np.column_stack([
            113.65 + 1.7461 * T - 0.28297 * T**2,
            -475.47 + 6.822 * T - 0.41443 * T**2,
            91.459 - 12.848 * T + 0.53951 * T**2
        ])

    elif sensor == 690:
        print("\nUsing sensor 0690")
        b = np.column_stack([
            -58.421 - 1.958 * T + 6.1023e-2 * T**2,
            41.01 - 8.5493 * T + 3.0701e-1 * T**2,
            227.85 + 4.0855 * T - 1.8867e-1 * T**2
        ])
        scale = np.column_stack([
            0.99381 + 2.4977e-4 * T + 2.7546e-6 * T**2,
            0.98771 + 6.3261e-4 * T - 5.0385e-6 * T**2,
            0.99634 + 1.2419e-4 * T + 5.662e-6 * T**2
        ])
        r = np.column_stack([
            435.02 - 9.6061 * T + 0.20449 * T**2,
            193.36 + 6.2856 * T + 0.1021 * T**2,
            -189.36 + 2.2936 * T - 0.18431 * T**2
        ])

    # ---------------------------------------------------------------------
    # Step 3: Initialize outputs
    fraw = np.zeros(len(raw))
    fcor = np.zeros(len(raw))
    COR = np.zeros_like(raw)

    # ---------------------------------------------------------------------
    # Step 4: Apply scale, bias, and orientation correction
    for k in range(len(raw)):
        S = np.diag(scale[k, :])
        Sinv = np.linalg.inv(S)

        # Rotation angles in radians
        u = (r[k, :] / 3600.0) * np.pi / 180.0
        su = np.sin(u)
        cu = np.cos(u)
        w = np.sqrt(1 - su[1]**2 - su[2]**2)

        # Transformation matrix P
        P = np.zeros((3, 3))
        P[0, 0] = 1
        P[1, 0] = su[0] / cu[0]
        P[1, 1] = 1 / cu[0]
        P[2, 0] = -((su[0] * su[2] + cu[0] * su[1]) / (w * cu[0]))
        P[2, 1] = -su[2] / (w * cu[0])
        P[2, 2] = 1 / w

        PS = np.dot(P, Sinv)
        tmp = raw[k, :] - b[k, :]

        # Apply full correction
        cor = np.zeros(3)
        for i in range(3):
            for j in range(3):
                cor[i] += PS[i, j] * tmp[j]

        fraw[k] = np.sqrt(np.sum(raw[k, :]**2))
        fcor[k] = np.sqrt(np.sum(cor**2))
        COR[k, :] = cor

    fcorT = fcor.copy()
    print("Performed")
    print("-------- sensor calibration")
    print("-------- temperature correction")
    print("-------- component rotation\n")

    return COR, fcorT

def Mag_temperature_corr(sensor_number, mag_x, mag_y, mag_z, mag_temp, nbr_ind):
    """
    Temperature correction to magnetometer data using CALIB1520 standard.

    Args:
        sensor_number (int): Sensor ID (e.g., 690 for top fluxgate)
        mag_x (np.ndarray): Raw magnetic field measurements, X component
        mag_y (np.ndarray): Raw magnetic field measurements, Y component
        mag_z (np.ndarray): Raw magnetic field measurements, Z component
        mag_temp (np.ndarray): Temperature measurements
        nbr_ind (np.ndarray or list): Indices of samples to process

    Returns:
        XYZ_tcal (np.ndarray): Nx3 matrix of temperature-corrected magnetic data [X, Y, Z]
        mag_tcal (np.ndarray): Full output from mag_calib1520_sensors (includes all corrections)
    """
    # ---------------------------------------------------------------------
    # Step 1: Extract data for the selected indices and stack into Nx4 matrix
    mag_xt = mag_x[nbr_ind]
    mag_yt = mag_y[nbr_ind]
    mag_zt = mag_z[nbr_ind]
    mag_tempt = mag_temp[nbr_ind]

    mag_xyzt = np.column_stack((mag_xt, mag_yt, mag_zt, mag_tempt))

    # Call the sensor calibration function
    XYZ_tcal, mag_tcal = Mag_calib1520_sensors(mag_xyzt, sensor_number)

    return XYZ_tcal, mag_tcal

def Mag_component_rotation(heading, pitch, roll, XYZ_tcal, orientation):
    """
    Rotate calibrated XYZ magnetic field measurements into NED frame.

    Parameters
    ----------
    heading : array-like
        Heading in degrees
    pitch : array-like
        Pitch in degrees
    roll : array-like
        Roll in degrees
    XYZ_tcal : ndarray, shape (N, 3)
        Temperature-corrected magnetic field components [X, Y, Z]
    orientation : int
        Sensor mounting orientation:
            0: X =  X, Y =  Y, Z =  Z
            1: X = -X, Y = -Y, Z =  Z (top mount)
            2: X = -X, Y =  Y, Z = -Z (port mount)
            3: X =  X, Y = -Y, Z = -Z (starboard mount)
            4: X = -X, Y =  Y, Z =  Z
            5: X =  X, Y =  Y, Z = -Z

    Returns
    -------
    mag_ned : ndarray, shape (N, 3)
        Rotated magnetic field components in NED frame (North, East, Down)
    """

    # Ensure inputs are numpy arrays
    heading = np.asarray(heading, dtype=float)
    pitch = np.asarray(pitch, dtype=float)
    roll = np.asarray(roll, dtype=float)
    XYZ_tcal = np.asarray(XYZ_tcal, dtype=float)

    # -------------------------------------------------------------------------
    # Step 1: Convert degrees to radians
    alpha = np.deg2rad(heading)  # Heading
    beta = np.deg2rad(pitch)     # Pitch
    gamma = np.deg2rad(roll)     # Roll

    mag_ned = np.zeros_like(XYZ_tcal)

    # -------------------------------------------------------------------------
    # Step 2: Adjust sensor axes based on mounting orientation
    if orientation == 0:
        pass  # No changes needed

    elif orientation == 1:
        print("\nComponent rotation for top maggie")
        XYZ_tcal[:, 0] = -XYZ_tcal[:, 0]  # X
        XYZ_tcal[:, 1] = -XYZ_tcal[:, 1]  # Y

    elif orientation == 2:
        print("\nComponent rotation for port maggie")
        XYZ_tcal[:, 0] = -XYZ_tcal[:, 0]  # X
        XYZ_tcal[:, 2] = -XYZ_tcal[:, 2]  # Z

    elif orientation == 3:
        print("\nComponent rotation for starboard maggie")
        XYZ_tcal[:, 1] = -XYZ_tcal[:, 1]  # Y
        XYZ_tcal[:, 2] = -XYZ_tcal[:, 2]  # Z

    elif orientation == 4:
        XYZ_tcal[:, 0] = -XYZ_tcal[:, 0]  # X

    elif orientation == 5:
        XYZ_tcal[:, 2] = -XYZ_tcal[:, 2]  # Z

    else:
        raise ValueError(f"Unknown orientation code: {orientation}")

    # -------------------------------------------------------------------------
    # Step 3: Rotate each sample to NED
    n_samples = min(len(heading), XYZ_tcal.shape[0])
    for k in range(n_samples):
        sa = np.sin(alpha[k])
        sb = np.sin(beta[k])
        sg = np.sin(gamma[k])
        ca = np.cos(alpha[k])
        cb = np.cos(beta[k])
        cg = np.cos(gamma[k])

        # Build rotation matrix T
        T = np.zeros((3, 3))
        T[0, 0] = ca * cb
        T[1, 0] = ca * sb * sg - sa * cg
        T[2, 0] = ca * sb * cg + sa * sg
        T[0, 1] = sa * cb
        T[1, 1] = sa * sb * sg + ca * cg
        T[2, 1] = sa * sb * cg - ca * sg
        T[0, 2] = -sb
        T[1, 2] = cb * sg
        T[2, 2] = cb * cg

        # Rotate calibrated XYZ vector into NED
        mag_ned[k, :] = (T.T @ XYZ_tcal[k, :])

    return mag_ned

def sine_model(x, a, b, c, d, e):
    """
    a + b*sind(c + x) + d*sind(e + 2*x)
    All angles in degrees
    """
    return (
        a
        + b * np.sin(np.radians(c + x))
        + d * np.sin(np.radians(e + 2 * x))
    )

def robust_curve_fit(x, y, initial_guess):
    # First fit
    popt, _ = curve_fit(
        sine_model,
        x,
        y,
        p0=initial_guess,
        maxfev=10000
    )

    # Residuals and outlier rejection
    residuals = y - sine_model(x, *popt)
    mask = np.abs(residuals) < 2 * np.std(residuals)

    # Refit without outliers
    popt_robust, _ = curve_fit(
        sine_model,
        x[mask],
        y[mask],
        p0=popt,
        maxfev=10000
    )

    return popt_robust

def Mag_fit_spin_comp(heading, n_comp, e_comp, d_comp, sentry_number=None):
    """
    Fit robust sine models to spin maneuver components.
    Plots are saved automatically and figures are closed to prevent hanging.
    """
    heading = np.asarray(heading)
    n_comp = n_comp - np.mean(n_comp)
    e_comp = e_comp - np.mean(e_comp)
    d_comp = d_comp - np.mean(d_comp)
	
	# ---------------------------------------------------------------------
    # Step 1: Fit robust sine models
    popt_n = robust_curve_fit(heading, n_comp, [0, np.std(n_comp), 0, np.std(n_comp)/4, 0])
    popt_e = robust_curve_fit(heading, e_comp, [0, np.std(e_comp), 0, np.std(e_comp)/4, 0])
    popt_d = robust_curve_fit(heading, d_comp, [0, np.std(d_comp), 0, np.std(d_comp)/4, 0])

    headings_plot = np.linspace(0, 360, 720)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    comps = [(n_comp, popt_n, 'North'), (e_comp, popt_e, 'East'), (d_comp, popt_d, 'Down')]

    for ax, (data, popt, label) in zip(axes, comps):
        ax.plot(heading, data, 'k.', markersize=2, alpha=0.6, label='Data')
        ax.plot(headings_plot, sine_model(headings_plot, *popt), 'm-', linewidth=2, label='Robust sin curve')
        ylim_val = np.max(np.abs(data))
        ax.set_ylim([-ylim_val, ylim_val])
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xticklabels(['0','90','180','270','360'])
        ax.set_title(label)
        ax.set_xlabel('Heading (°)')
        if label == 'North':
            ax.set_ylabel('Magnetic Field (nT)')
        ax.grid(True)
        if label == 'North':
            ax.legend(loc='lower right')
        else:
            ax.legend().set_visible(False)

    plt.tight_layout()

    # Save and close figure
    if sentry_number is not None:
        plt.savefig(f'sentry_{sentry_number}_spin_fit.png', dpi=200)
    plt.close(fig)

    # Return first 3 parameters of each component (like MATLAB)
    Fit_Matrix = np.vstack([popt_n[:3], popt_e[:3], popt_d[:3]])
    return Fit_Matrix

def Mag_plot_filter_components(time, x_comp, y_comp, z_comp, win_size, 
                                title1, title2, title3, filter_type=0):
    """
    Visualize 3-axis magnetic data with optional filtering.
    
    Parameters:
        time       : array-like, time values
        x_comp     : array-like, X component of magnetic field
        y_comp     : array-like, Y component
        z_comp     : array-like, Z component
        win_size   : int, window size for filtering (will be forced to odd)
        title1-3   : str, titles for subplots
        filter_type: int
            0 = No filter,
            1 = Median filter,
            2 = Median + Savitzky-Golay smoothing
    Returns:
        Filtered : np.ndarray of shape (N, 3), filtered components
    """
    # Ensure inputs are numpy arrays
    time = np.array(time)
    x_comp = np.array(x_comp)
    y_comp = np.array(y_comp)
    z_comp = np.array(z_comp)

    # Ensure window size is odd and >= 3
    if win_size % 2 == 0:
        win_size += 1
    if win_size < 3:
        win_size = 3

    Filtered = np.vstack([x_comp, y_comp, z_comp]).T  # default: raw data

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    titles = [title1, title2, title3]
    comps_raw = [x_comp, y_comp, z_comp]

    if filter_type == 1:
        # Moving median
        x_med = medfilt(x_comp, win_size)
        y_med = medfilt(y_comp, win_size)
        z_med = medfilt(z_comp, win_size)
        Filtered = np.vstack([x_med, y_med, z_med]).T

        for i, (ax, comp, med, title) in enumerate(zip(axes, comps_raw, [x_med, y_med, z_med], titles)):
            ax.plot(time, comp, 'k-', linewidth=0.5, label='Raw')
            ax.plot(time, med, 'm-', linewidth=1.5, label='Median')
            ax.set_title(title)
            ax.grid(True)
            if i == 1:
                ax.set_ylabel('Magnetic Field (nT)')
            if i == 2:
                ax.set_xlabel('Time')
            ax.legend(loc='upper right')

    elif filter_type == 2:
        # Median filter first
        x_med = medfilt(x_comp, win_size)
        y_med = medfilt(y_comp, win_size)
        z_med = medfilt(z_comp, win_size)

        # Savitzky-Golay smoothing (must use odd window, polyorder < win_size)
        sg_win = win_size
        polyorder = min(3, sg_win - 1)

        x_smooth = savgol_filter(x_med, sg_win, polyorder)
        y_smooth = savgol_filter(y_med, sg_win, polyorder)
        z_smooth = savgol_filter(z_med, sg_win, polyorder)
        Filtered = np.vstack([x_smooth, y_smooth, z_smooth]).T

        for i, (ax, comp, med, smooth, title) in enumerate(
            zip(axes, comps_raw, [x_med, y_med, z_med], [x_smooth, y_smooth, z_smooth], titles)
        ):
            ax.plot(time, comp, 'k-', linewidth=0.5, label='Raw')
            ax.plot(time, med, 'm-', linewidth=1.5, label='Median')
            ax.plot(time, smooth, 'b-', linewidth=1.5, label='L2 (SG)')
            ax.set_title(title)
            ax.grid(True)
            if i == 1:
                ax.set_ylabel('Magnetic Field (nT)')
            if i == 2:
                ax.set_xlabel('Time')
            ax.legend(loc='upper right')

    else:
        # No filter
        for i, (ax, comp, title) in enumerate(zip(axes, comps_raw, titles)):
            ax.plot(time, comp, 'k-', linewidth=0.5, label='Raw')
            ax.set_title(title)
            ax.grid(True)
            if i == 1:
                ax.set_ylabel('Magnetic Field (nT)')
            if i == 2:
                ax.set_xlabel('Time')
            ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    return Filtered

def Mag_spin_fitting(mag, sbe, spinFitFile, spinTimeFile, sentry_number):
    """
    Performs spin fitting calibration for a magnetometer survey.
    All figures are saved automatically, no blocking.
    """
    print(f"Starting spin fitting for Sentry {sentry_number}...")

    # Select spin maneuver times interactively
    spin_start, spin_end = Mag_find_spins(
        sbe['heading'], sbe['lon'], sbe['lat'], sbe['utime'], sbe['alt']
    )

    # ---------------------------------------------------------------------
    # Step 1: Select magnetometer indices for spin
    idx = (mag['utime'] >= spin_start) & (mag['utime'] <= spin_end)
    mag_ut = mag['utime'][idx]
    mag_dt = [datetime.fromtimestamp(float(t), tz=timezone.utc) for t in mag_ut]

    mag_x = mag['x'][idx] * 1e5
    mag_y = mag['y'][idx] * 1e5
    mag_z = mag['z'][idx] * 1e5

    hdg   = mag['heading'][idx]
    pitch = mag['pitch'][idx]
    roll  = mag['roll'][idx]

    # ---------------------------------------------------------------------
    # Step 2: Plot raw & filtered components
    win = round(0.1 * (mag['samp_rate'] / 0.0166667))

    Filtered = Mag_plot_filter_components(
        mag_dt, mag_x, mag_y, mag_z,
        win, 'Mag X', 'Mag Y', 'Mag Z', filter_type=1
    )
    if sentry_number is not None:
        plt.savefig(f'sentry_{sentry_number}_mag_filtered.png', dpi=200)
        plt.close('all')

    # ---------------------------------------------------------------------
    # Step 3: Temperature calibration
    sensor = 690
    XYZ_tcal, _ = Mag_temperature_corr(
        sensor,
        mag['x'], mag['y'], mag['z'],
        mag['temp'],
        np.where(idx)[0]
    )

    # Component rotation to NED
    ned = Mag_component_rotation(hdg, pitch, roll, XYZ_tcal, orientation=0)
    n, e, d = ned[:, 0], ned[:, 1], ned[:, 2]

    # Plot filtered NED
    Filtered_NED = Mag_plot_filter_components(
        mag_dt, n, e, d,
        win, 'North', 'East', 'Down', filter_type=1
    )
    if sentry_number is not None:
        plt.savefig(f'sentry_{sentry_number}_ned_filtered.png', dpi=200)
        plt.close('all')

    # ---------------------------------------------------------------------
    # Step 4: Spin fit
    Fitting = Mag_fit_spin_comp(hdg, n, e, d, sentry_number=sentry_number)

    print("Spin calibration and corresponding field components plots are saved")
    print("#--------------- Spin calibration complete ---------------#")

    # ---------------------------------------------------------------------
    # Step 5: Save outputs (MATLAB-compatible)
    sio.savemat(spinFitFile, {'Fitting': Fitting})
    sio.savemat(spinTimeFile, {'spin_time': np.array([spin_start, spin_end])})

    return Fitting

def Mag_find_straight_segment(duration, sbe_utime, sbe_lon, sbe_lat, mag_utime):
    """
    Extract the straight segment of a dive track.

    Parameters
    ----------
    duration : float
        Duration of straight segment to extract (seconds)
    sbe_utime : array-like
        Unix time vector from SBE
    sbe_lon : array-like
        Longitude from Sentry navigation
    sbe_lat : array-like
        Latitude from Sentry navigation
    mag_utime : array-like
        Unix time vector from magnetometer (used for sampling rate)

    Returns
    -------
    start_utime : float
        Start Unix time of selected straight segment
    end_utime : float
        End Unix time of selected straight segment
    """

    sbe_utime = np.asarray(sbe_utime)
    sbe_lon = np.asarray(sbe_lon)
    sbe_lat = np.asarray(sbe_lat)
    mag_utime = np.asarray(mag_utime)

    # -------------------------------------------------------------------------
    # Step 1: Compute magnetometer sampling rate
    mag_samp_rate = round(1 / ((mag_utime[-1] - mag_utime[0]) / len(mag_utime)))
    n_samples = round(duration / mag_samp_rate)

    # Midpoint of the dive
    sbe_utime_mid = sbe_utime[0] + (sbe_utime[-1] - sbe_utime[0]) / 2

    # -------------------------------------------------------------------------
    # Step 2: Find segment with maximum R^2 (linear regression)
    rsq_max = -np.inf
    sbe_utime_best = sbe_utime_mid

    # Slide the window forward three times (like MATLAB loop)
    for _ in range(3):
        # Indices within the current window
        FG_ind = np.where(np.abs(mag_utime - sbe_utime_mid) < duration / 2)[0]

        if len(FG_ind) > 1:
            # Linear regression: lat vs lon
            X = sbe_lon[FG_ind].reshape(-1, 1)
            y = sbe_lat[FG_ind]
            model = LinearRegression().fit(X, y)
            rsq = model.score(X, y)  # R^2 value

            if rsq > rsq_max:
                rsq_max = rsq
                sbe_utime_best = sbe_utime_mid

        # Move window forward
        sbe_utime_mid += duration

    # -------------------------------------------------------------------------
    # Step 3: Plot the segment
    plt.figure(figsize=(6,6))
    plt.plot(sbe_lon, sbe_lat, 'b-', linewidth=0.8, label='Full Dive Track')

    # Indices of best segment
    mag_ind = np.where(np.abs(sbe_utime - sbe_utime_best) < duration / 2)[0]
    plt.plot(sbe_lon[mag_ind], sbe_lat[mag_ind], 'r-', linewidth=3, label='Straight Segment')

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f'Best Straight Segment ({duration:.0f} s) - R² = {rsq_max:.3f}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------------------------------------------------------------------------
    # Step 4: Compute start and end Unix times
    start_utime = sbe_utime_best - duration / 2
    end_utime = sbe_utime_best + duration / 2

    return start_utime, end_utime

def Mag_process_segment(mag, sbe, Fitting):
    """
    Process a straight-line segment of magnetic data.

    Parameters:
        mag : dict - Magnetometer data
        sbe : dict - SBE data (used to find straight segments)
        Fitting : np.ndarray - 3x3 matrix of spin correction coefficients
    """

    print("========== Segment Processing ==========")
    duration = float(input("Segment duration (s)? "))
    seg_start, seg_end = Mag_find_straight_segment(duration, sbe['utime'], sbe['lon'], sbe['lat'], mag['utime'])
    
    idx = (mag['utime'] >= seg_start) & (mag['utime'] <= seg_end)

    # -- Extract variables --
    utime = mag['utime'][idx]
    mag_dt = [datetime.utcfromtimestamp(t) for t in utime]

    mag_x = mag['x'][idx] * 1e5
    mag_y = mag['y'][idx] * 1e5
    mag_z = mag['z'][idx] * 1e5

    hdg = mag['heading'][idx]
    pitch = mag['pitch'][idx]
    roll = mag['roll'][idx]

    win = round(5 * (mag['samp_rate'] / 0.0166667))

    # -------------------------
    # First plot: raw components
    Mag_plot_filter_components(mag_dt, mag_x, mag_y, mag_z, win, 'Mag X', 'Mag Y', 'Mag Z')
    plt.show(block=True)
    plt.close()        # Close figure to free memory

    # -- Temperature correction --
    sensor = 690
    XYZ_tcal, _ = Mag_temperature_corr(sensor, mag['x'], mag['y'], mag['z'], mag['temp'], np.where(idx)[0])

    # -- Component rotation --
    ned = Mag_component_rotation(hdg, pitch, roll, XYZ_tcal, orientation=0)
    n, e, d = ned[:, 0], ned[:, 1], ned[:, 2]

    # -------------------------
    # Second plot: NED components
    Mag_plot_filter_components(mag_dt, n, e, d, win, 'North', 'East', 'Down')
    plt.show(block=True)
    plt.close()

    # -- Heading correction --
    Fitting = Fitting.flatten()
    corn = Fitting[0] + Fitting[3] * np.sin(np.radians(hdg + Fitting[6]))
    core = Fitting[1] + Fitting[4] * np.sin(np.radians(hdg + Fitting[7]))
    corz = Fitting[2] + Fitting[5] * np.sin(np.radians(hdg + Fitting[8]))

    Nc = n - corn
    Ec = e - core
    Dc = d - corz

    # -------------------------
    # Third plot: corrected NED
    Mag_plot_filter_components(mag_dt, Nc, Ec, Dc, win, 'North-corrected', 'East-corrected', 'Down-corrected')
    plt.show(block=True)
    plt.close()

    # -------------------------
    # Remaining calculations (IGRF, anomalies, etc.)
    lat1 = mag['lat'][np.where(idx)[0][0]]
    lon1 = mag['lon'][np.where(idx)[0][0]]
    Be, Bn, Bu = ppigrf.igrf(lon1, lat1, 0.003, datetime(2024,12,4))
    X = Bn
    Y = Be
    Z = -Bu
    F = (X**2 + Y**2 + Z**2)**0.5
    IGRF = [X, Y, Z, F]

    Dec = 11.0707
    Inc = -21.5330

    anom_n = Nc - IGRF[0]
    anom_e = Ec - IGRF[1]
    anom_d = Dc - IGRF[2]

    anom_t = (anom_n * np.cos(np.radians(Dec)) * np.cos(np.radians(Inc)) +
              anom_e * np.sin(np.radians(Dec)) * np.cos(np.radians(Inc)) +
              anom_d * np.sin(np.radians(Inc)))

    # -------------------------
    # Fourth plot: anomaly (optional, can leave open)
    Mag_plot_filter_components(mag_dt, anom_n, anom_e, anom_d, win,
                               'North-Anomaly', 'East-Anomaly', 'Down-Anomaly')
    plt.show(block=True)
    plt.close()

def Mag_select_dive_segment(sbe_heading, sbe_lon, sbe_lat, sbe_utime, sbe_alt):
    """
    Identify and select a dive track segment interactively (like MATLAB version).

    Workflow:
        1. Display three figures: track, altitude vs time, heading vs time.
        2. User zooms Figure 3 to target region, then presses any key.
        3. User clicks two points on Figure 3 to select start and end of segment.
        4. Figures 1-3 are closed; segment plotted in Figure 4.

    Returns:
        seg_start : Unix time marking start of selected track
        seg_end   : Unix time marking end of selected track
    """

	# ---------------------------------------------------------------------
    # Step 1: Identify heading jumps
    ang = np.diff(sbe_heading)
    ang = np.append(ang, 0)
    ndc = np.where(np.abs(ang) > 180)[0]  # indices of large jumps

	# ---------------------------------------------------------------------
    # Step 2: Plot figures
    fig1, ax1 = plt.subplots()
    ax1.plot(sbe_lon, sbe_lat, 'b-', linewidth=0.8, label='Track')
    ax1.plot(sbe_lon[ndc], sbe_lat[ndc], 'ro', label='Heading change')
    ax1.set_title('Track with Heading Jumps')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend()

    fig2, ax2 = plt.subplots()
    ax2.plot(sbe_utime, sbe_alt, 'b-', linewidth=0.8)
    ax2.plot(sbe_utime[ndc], sbe_alt[ndc], 'ro')
    ax2.set_title('Altitude vs Time')
    ax2.set_xlabel('Unix Time')
    ax2.set_ylabel('Altitude (m)')

    fig3, ax3 = plt.subplots()
    ax3.plot(sbe_utime, sbe_heading, 'b-', linewidth=0.8)
    ax3.plot(sbe_utime[ndc], sbe_heading[ndc], 'ro')
    ax3.set_title('Heading vs Time')
    ax3.set_xlabel('Unix Time')
    ax3.set_ylabel('Heading (deg)')

    plt.tight_layout()
    plt.ion()
    plt.show(block=False)

    print("\nZoom/pan Figure 3 to the segment, then press any key to select start/end points...")

    selected_segment = []

    def on_key(event):
        """Triggered after key press to allow user to select two points for segment limits."""
        fig3.canvas.mpl_disconnect(cid_key)
        print("Click two points on Figure 3 to define segment limits...")
        clicked_pts = plt.ginput(2, timeout=-1)
        x1, x2 = clicked_pts[0][0], clicked_pts[1][0]

        seg_start = min(x1, x2)
        seg_end = max(x1, x2)
        print(f"Selected segment: start={seg_start:.2f}, end={seg_end:.2f}")

        # Zoom other figures to selected range
        ax3.set_xlim(seg_start, seg_end)
        ax2.set_xlim(seg_start, seg_end)

        # Plot selected dive track in new figure
        fig4, ax4 = plt.subplots()
        track_ndc = np.where((sbe_utime > seg_start) & (sbe_utime < seg_end))[0]
        ax4.plot(sbe_lon, sbe_lat, 'b-', linewidth=0.8)
        ax4.plot(sbe_lon[track_ndc], sbe_lat[track_ndc], 'r-', linewidth=1.5)
        ax4.set_title('Selected Dive Track Segment')
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        plt.show()

        # Store selected segment
        selected_segment.append((seg_start, seg_end))

        # Close figures 1-3 to avoid hanging
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        # fig4 stays open for inspection

    # Connect key press event
    cid_key = fig3.canvas.mpl_connect('key_press_event', on_key)

    plt.show(block=True)  # Wait for key press and clicks

    if selected_segment:
        return selected_segment[0]
    else:
        return None, None

def Mag_process_dive_without_spins(mag, sbe, Fitting, file_prefix):
    """
    Process and save magnetic data excluding spins.
    Applies heading corrections, filters, computes anomaly,
    saves to .mat and .txt files.

    Parameters
    ----------
    mag : dict
        Magnetometer data with keys: utime, x, y, z, heading, pitch, roll, temp, lat, lon, samp_rate
    sbe : dict
        SBE data with keys: depth, temp, alt, heading, lon, lat, utime, samp_rate
    Fitting : array-like
        Spin fitting coefficients
    file_prefix : str
        File name prefix for outputs
    """

    matFile = f'Seg_{file_prefix}.mat'
    txtFile = f'Seg_{file_prefix}.txt'

	# ---------------------------------------------------------------------
    # Step 1: Check if file exists
    if os.path.isfile(matFile):
        print(f'File already exists: {matFile}')
        choice = input('Recompute anyway? (y/n): ').strip().lower()
        if choice != 'y':
            print('Check for already saved file.')
            return
        else:
            print('Recomputing as requested.')

    # -------------------------------------------------------------------------
    # Step 2: segment selection
    seg_start, seg_end = Mag_select_dive_segment(sbe['heading'], sbe['lon'], sbe['lat'], sbe['utime'], sbe['alt'])
    idx = (mag['utime'] >= seg_start) & (mag['utime'] <= seg_end)
    N = np.sum(idx)

    # -------------------------------------------------------------------------
    # Step 3: compute variables
    mag_ut = mag['utime'][idx]
    mag_dt = [datetime.fromtimestamp(t, tz=timezone.utc) for t in mag_ut]
    hdg = mag['heading'][idx]
    pitch = mag['pitch'][idx]
    roll = mag['roll'][idx]

    # Magnetometer data (nT)
    mag_x = mag['x'][idx] * 1e5
    mag_y = mag['y'][idx] * 1e5
    mag_z = mag['z'][idx] * 1e5

    # -------------------------------------------------------------------------
    # Step 4: data processing
    XYZ_tcal, _ = Mag_temperature_corr(690, mag['x'], mag['y'], mag['z'], mag['temp'], np.where(idx)[0])
    
    ned = Mag_component_rotation(hdg, pitch, roll, XYZ_tcal, 0)
    n, e, d = ned[:,0], ned[:,1], ned[:,2]

    # Heading-dependent correction
    Fitting = Fitting.flatten()
    corn = Fitting[0] + Fitting[3] * np.sin(np.deg2rad(hdg + Fitting[6]))
    core = Fitting[1] + Fitting[4] * np.sin(np.deg2rad(hdg + Fitting[7]))
    corz = Fitting[2] + Fitting[5] * np.sin(np.deg2rad(hdg + Fitting[8]))

    Nc = n - corn
    Ec = e - core
    Dc = d - corz

    # Filtering
    win = 3 * round(mag['samp_rate'] / 0.0166667)
    Filtered = Mag_plot_filter_components(mag_dt, Nc, Ec, Dc, win, 'North-corrected', 'East-corrected', 'Down-corrected', 1)

    # Field strengths
    mag_total_corr = np.sqrt(np.sum(Filtered**2, axis=1))
    mag_total_orig = np.sqrt(n**2 + e**2 + d**2)

    # IGRF model
    lat1 = mag['lat'][np.where(idx)[0][0]]
    lon1 = mag['lon'][np.where(idx)[0][0]]
    Be, Bn, Bu = ppigrf.igrf(lon1, lat1, 0.003, datetime(2024,12,4))
    X = Bn
    Y = Be
    Z = -Bu
    F = (X**2 + Y**2 + Z**2)**0.5
    IGRF = [X, Y, Z, F]
    
    # Declination & Inclination
    Dec = 11.0707
    Inc = -21.5330

    # Magnetic anomaly
    anom = ((Filtered[:,0]-IGRF[0])*np.cos(np.deg2rad(Dec))*np.cos(np.deg2rad(Inc)) +
            (Filtered[:,1]-IGRF[1])*np.sin(np.deg2rad(Dec))*np.cos(np.deg2rad(Inc)) +
            (Filtered[:,2]-IGRF[2])*np.sin(np.deg2rad(Inc)))

    # -------------------------------------------------------------------------
    # Step 5: Resample SBE data to exactly match mag_ut
    sbe_idx = (sbe['utime'] >= seg_start) & (sbe['utime'] <= seg_end)
    sbe_ut_seg = sbe['utime'][sbe_idx]
    sbe_depth_seg = sbe['depth'][sbe_idx]
    sbe_temp_seg  = sbe['temp'][sbe_idx]
    sbe_alt_seg   = sbe['alt'][sbe_idx]

    sbe_depth1 = np.interp(mag_ut, sbe_ut_seg, sbe_depth_seg)
    sbe_temp1  = np.interp(mag_ut, sbe_ut_seg, sbe_temp_seg)
    sbe_alt1   = np.interp(mag_ut, sbe_ut_seg, sbe_alt_seg)

    # -------------------------------------------------------------------------
    # Step 6: save files
    output = np.column_stack([
        mag_ut, mag['lat'][idx], mag['lon'][idx],
        sbe_depth1, sbe_temp1, sbe_alt1,
        hdg, roll, pitch,
        mag_x, mag_y, mag_z,
        n, e, d,
        mag_total_orig,
        Filtered[:,0], Filtered[:,1], Filtered[:,2],
        mag_total_corr,
        Filtered[:,0]-IGRF[0], Filtered[:,1]-IGRF[1], Filtered[:,2]-IGRF[2],
        anom
    ])

    header = ('utime,lat,lon,Depth,Temp,Altitude,Heading,Roll,Pitch,mag_x,mag_y,mag_z,'
              'N,E,D,mag_total_orig,N_corr,E_corr,D_corr,mag_total_corr,Anom_n,Anom_e,Anom_d,Anom_total')

    # Downsample per second (MATLAB downsample = take every 6th sample)
    output = output[::6]

    # Save .mat
    mag_top = {
        'utime': output[:,0],
        'lat': output[:,1],
        'lon': output[:,2],
        'depth': output[:,3],
        'temp': output[:,4],
        'alt': output[:,5],
        'heading': output[:,6],
        'roll': output[:,7],
        'pitch': output[:,8],
        'mag_x': output[:,9],
        'mag_y': output[:,10],
        'mag_z': output[:,11],
        'n': output[:,12],
        'e': output[:,13],
        'd': output[:,14],
        'mag_total_orig': output[:,15],
        'N_corr': output[:,16],
        'E_corr': output[:,17],
        'D_corr': output[:,18],
        'mag_total_corr': output[:,19],
        'anom_n': output[:,20],
        'anom_e': output[:,21],
        'anom_d': output[:,22],
        'anom_total': output[:,23]
    }

    scipy.io.savemat(matFile, {'mag_top': mag_top})

    # Save .txt
    np.savetxt(txtFile, output, delimiter=',', header=header, comments='', fmt='%.6f')

    print(f'\nDive data saved to {matFile} and \n                   {txtFile}\n')

def Mag_save_nospins(magData, sbeData, Fitting, file_prefix, output_file):
    """
    Processes and optionally saves dive data excluding spin times.

    - Checks if the specified output file already exists
    - If it does, prompts the user whether to recompute the data
    - If not or if the user chooses to recompute, processes dive data 
      while excluding spin times

    Parameters:
        magData      : Data structure for magnetometer readings
        sbeData      : Data structure for SBE (e.g., CTD) readings
        Fitting      : Fitting parameters or model coefficients
        file_prefix  : String prefix for intermediate/output files
        output_file  : Path to the output file
    """
    
    if os.path.isfile(output_file):
        prompt = f'File "{output_file}" exists. Recompute? (y/n): '
        if ask_yesno(prompt):
            print('Recomputing no-spin output...')
            Mag_process_dive_without_spins(magData, sbeData, Fitting, file_prefix)
        else:
            print('Skipped recomputation of dive without spins.')
    else:
        print('\nProcessing and saving dive excluding spin times...')
        Mag_process_dive_without_spins(magData, sbeData, Fitting, file_prefix)
        
def Mag_process_dive_with_spins(mag, sbe, Fitting, file_prefix):
    """
    Process magnetometer data with spin corrections, filtering, and field model correction.
    Saves output to .mat and .txt files at 1s intervals.
    
    Parameters
    ----------
    mag : dict
        Magnetometer data (utime, x, y, z, heading, pitch, roll, temp, lat, lon, samp_rate)
    sbe : dict
        SBE data (depth, temp, alt, samp_rate)
    Fitting : list or np.ndarray
        Instrument correction fitting parameters
    file_prefix : str
        Prefix for output filenames
    """ 
    
    matFile = f"Mag_data_{file_prefix}.mat"
    txtFile = f"Mag_data_{file_prefix}.txt"

    # -------------------------------------------------------------------------
    # Step 1: Check for existing file
    if os.path.isfile(matFile):
        choice = input(f"File already exists: {matFile}\nRecompute anyway? (y/n): ").strip().lower()
        if choice != 'y':
            print("Check for already saved file.")
            return
        else:
            print("Recomputing...........")

    # -------------------------------------------------------------------------
    # Step 2: Index mask and sizes
    idx = np.ones(len(mag['utime']), dtype=bool)
    N = np.sum(idx)

    # Time & orientation
    mag_ut = np.array(mag['utime'])[idx]
    mag_dt = [datetime.utcfromtimestamp(t) for t in mag_ut]
    hdg = np.array(mag['heading'])[idx]
    pitch = np.array(mag['pitch'])[idx]
    roll = np.array(mag['roll'])[idx]

    # Raw magnetometer data (Gauss → nT)
    mag_x = np.array(mag['x'])[idx] * 1e5
    mag_y = np.array(mag['y'])[idx] * 1e5
    mag_z = np.array(mag['z'])[idx] * 1e5

    # -------------------------------------------------------------------------
    # Step 3: Processing steps

    # Temperature correction
    XYZ_tcal, _ = Mag_temperature_corr(690, mag['x'], mag['y'], mag['z'], mag['temp'], np.where(idx)[0])

    # Coordinate rotation
    ned = Mag_component_rotation(hdg, pitch, roll, XYZ_tcal, 0)
    n, e, d = ned[:,0], ned[:,1], ned[:,2]

    # Heading-dependent instrument correction
    Fitting = Fitting.flatten()
    corn = Fitting[0] + Fitting[3] * np.sin(np.radians(hdg + Fitting[6]))
    core = Fitting[1] + Fitting[4] * np.sin(np.radians(hdg + Fitting[7]))
    corz = Fitting[2] + Fitting[5] * np.sin(np.radians(hdg + Fitting[8]))

    # Corrected field
    Nc = n - corn
    Ec = e - core
    Dc = d - corz

    # Filter components
    win = round(mag['samp_rate'] / 0.0166667)  # convert to minutes
    Filtered = Mag_plot_filter_components(mag_dt, Nc, Ec, Dc, win,
                                          'North-corrected', 'East-corrected', 'Down-corrected', 1)

    # Field strength
    mag_total_corr = np.sqrt(np.sum(Filtered**2, axis=1))
    mag_total_orig = np.sqrt(n**2 + e**2 + d**2)

    # Compute IGRF model
    lat1 = mag['lat'][np.where(idx)[0][0]]
    lon1 = mag['lon'][np.where(idx)[0][0]]
    Be, Bn, Bu = ppigrf.igrf(lon1, lat1, 0.003, datetime(2024,12,4))
    X = Bn
    Y = Be
    Z = -Bu
    F = (X**2 + Y**2 + Z**2)**0.5
    IGRF = [X, Y, Z, F]

    # Decl & Incl (hardcoded)
    Dec, Inc = 11.0707, -21.5330

    # Magnetic anomaly
    anom = ((Filtered[:,0] - IGRF[0]) * np.cos(np.radians(Dec)) * np.cos(np.radians(Inc)) +
            (Filtered[:,1] - IGRF[1]) * np.sin(np.radians(Dec)) * np.cos(np.radians(Inc)) +
            (Filtered[:,2] - IGRF[2]) * np.sin(np.radians(Inc)))

    # -------------------------------------------------------------------------
    # Step 4: Resample SBE data
    sbe_depth1 = resample(sbe['depth'], N)
    sbe_temp1 = resample(sbe['temp'], N)
    sbe_alt1 = resample(sbe['alt'], N)

    # -------------------------------------------------------------------------
    # Step 5: Build output array
    output = np.column_stack([
        mag_ut, mag['lat'][idx], mag['lon'][idx],
        sbe_depth1, sbe_temp1, sbe_alt1,
        hdg, roll, pitch,
        mag_x, mag_y, mag_z,
        n, e, d,
        mag_total_orig,
        Filtered[:,0], Filtered[:,1], Filtered[:,2],
        mag_total_corr,
        Filtered[:,0] - IGRF[0], Filtered[:,1] - IGRF[1], Filtered[:,2] - IGRF[2],
        anom
    ])

    header = [
        'utime','lat','lon','Depth','Temp','Altitude','Heading','Roll','Pitch',
        'mag_x','mag_y','mag_z','N','E','D','mag_total_orig',
        'N_corr','E_corr','D_corr','mag_total_corr',
        'Anom_n','Anom_e','Anom_d','Anom_total'
    ]

    # Downsample to 1Hz (original sampling is 6 Hz)
    output = output[::6, :]

    # -------------------------------------------------------------------------
    # Step 6: Save files

    # Save MAT file
    mag_top = {name: output[:,i] for i, name in enumerate(header)}
    sio.savemat(matFile, {'mag_top': mag_top})

    # Save text file
    pd.DataFrame(output, columns=header).to_csv(txtFile, index=False, float_format="%.6f")

    print(f"\nDive data saved to {matFile} and \n                   {txtFile}\n")

def Mag_save_withspins(magData, sbeData, Fitting, file_prefix, output_file):
    """
    Processes and optionally saves dive data including spin times.

    - Checks if the specified output file already exists.
    - If it does, prompts the user whether to recompute the data.
    - If not or if the user chooses to recompute, processes dive data
      while including spin times.

    Parameters:
        magData     : Data structure or object containing magnetometer data.
        sbeData     : Data structure or object containing SBE (sensor) data.
        Fitting     : Dictionary or object containing fitting parameters.
        file_prefix : String used as a prefix for intermediate/output files.
        output_file : Full path to the output file to save processed data.
    """
    if os.path.isfile(output_file):
        if ask_yesno(f'File "{output_file}" exists. Recompute? (y/n): '):
            print('Recomputing with-spins output...')
            Mag_process_dive_with_spins(magData, sbeData, Fitting, file_prefix)
        else:
            print('Skipped recomputation of dive with spins.')
    else:
        print('\nProcessing and saving dive including spins time...')
        Mag_process_dive_with_spins(magData, sbeData, Fitting, file_prefix)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def Mag_crossover_correction(segment_length, heading_threshold, sentryNums):
    """
    Perform crossover correction on Sentry magnetic anomaly data.

    Parameters
    ----------
    segment_length : int
        Segment length (samples)
    heading_threshold : float
        Heading difference threshold (degrees)
    sentryNums : int or list[int]
        One or more sentry numbers
    """

    # Normalize sentryNums
    if isinstance(sentryNums, int):
        sentryNums = [sentryNums]

    segment_length = int(segment_length)

    # --------------------------------------------------------
    # Step 1: Discover files
    all_files = []
    for s in sentryNums:
        all_files.extend(glob.glob(f"Seg_sentry{s:03d}_*.txt"))

    pattern = re.compile(r"^Seg_sentry\d{3}_\d{8}_\d{4}\.txt$")
    fileList = sorted([f for f in all_files if pattern.match(os.path.basename(f))])

    if not fileList:
        raise RuntimeError("No valid Seg_sentry files found.")

    numDives = len(fileList)

    # Magnetic variables
    magVars = ["Anom_n", "Anom_e", "Anom_d", "Anom_total"]

    # --------------------------------------------------------
    # Step 2: Load all dives
    diveData = []
    for fname in fileList:
        df = pd.read_csv(fname)
        nseg = len(df) // segment_length
        diveData.append({
            "raw": df,
            "numSegments": nseg
        })

    # --------------------------------------------------------
    # Step 3: Loop over magnetic variables
    for magVar in magVars:
        print(f"\nProcessing {magVar}")

        segments = []
        segIdx = 0

        # ----------------------------------------------------
        # Step 4: Build segments
        for d, dive in enumerate(diveData):
            df = dive["raw"]
            N = dive["numSegments"]

            for i in range(N):
                idx = np.arange(segment_length) + i * segment_length

                segments.append({
                    "lat": df["lat"].values[idx],
                    "lon": df["lon"].values[idx],
                    "mag": df[magVar].values[idx],
                    "hdg": df["Heading"].values[idx],
                    "dive": d,
                    "localIdx": i
                })
                segIdx += 1

        numSegments = len(segments)

        # ----------------------------------------------------
        # Step 5: Find crossovers
        crossover_rows = []

        for i in range(numSegments - 1):
            for j in range(i + 1, numSegments):

                # Skip adjacent segments from same dive
                if (segments[i]["dive"] == segments[j]["dive"] and
                    abs(segments[i]["localIdx"] - segments[j]["localIdx"]) <= 1):
                    continue

                latA = segments[i]["lat"]
                lonA = segments[i]["lon"]
                magA = segments[i]["mag"]
                hdgA = segments[i]["hdg"]

                latB = segments[j]["lat"]
                lonB = segments[j]["lon"]
                magB = segments[j]["mag"]
                hdgB = segments[j]["hdg"]

                for k in range(len(latA)):
                    dists = haversine(latA[k], lonA[k], latB, lonB)
                    idxMin = np.argmin(dists)

                    if dists[idxMin] < 0.01:  # km
                        dh = abs(hdgA[k] - hdgB[idxMin])
                        dh = min(dh % 360, 360 - (dh % 360))

                        if dh > heading_threshold:
                            diffMag = magA[k] - magB[idxMin]
                            crossover_rows.append([i, j, diffMag])

        if not crossover_rows:
            print("  No valid crossovers found.")
            continue

        crossoverData = np.array(crossover_rows)

        # ----------------------------------------------------
        # Step 6: Least squares solution
        A = np.zeros((len(crossoverData), numSegments))
        b = np.zeros(len(crossoverData))

        for k, (i, j, diff) in enumerate(crossoverData):
            A[k, int(i)] = 1
            A[k, int(j)] = -1
            b[k] = -diff

        x, *_ = np.linalg.lstsq(A, b, rcond=None)

        # ----------------------------------------------------
        # Step 7: Apply corrections
        for i in range(numSegments):
            segments[i]["mag_corrected"] = segments[i]["mag"] + x[i]

        # ----------------------------------------------------
        # Step 8: Rebuild dives
        for d, dive in enumerate(diveData):
            df = dive["raw"].copy()
            N = dive["numSegments"]

            corrected = np.full(len(df), np.nan)

            for i in range(N):
                idx = np.arange(segment_length) + i * segment_length
                gIdx = next(
                    k for k, s in enumerate(segments)
                    if s["dive"] == d and s["localIdx"] == i
                )
                corrected[idx] = segments[gIdx]["mag_corrected"]

            df[f"{magVar}_crossover"] = corrected
            diveData[d]["raw"] = df

        # ----------------------------------------------------
        # Step 9: RMS statistics
        diff0 = crossoverData[:, 2]
        diff1 = diff0 + x[crossoverData[:, 0].astype(int)] - x[crossoverData[:, 1].astype(int)]

        print(f"  RMS before: {np.sqrt(np.mean(diff0**2)):.4f} nT")
        print(f"  RMS after : {np.sqrt(np.mean(diff1**2)):.4f} nT")

    # --------------------------------------------------------
    # Step 10: Save corrected files
    for d, fname in enumerate(fileList):
        df = diveData[d]["raw"]
        base, ext = os.path.splitext(fname)
        outFile = f"{base}_cross{ext}"

        df.to_csv(outFile, index=False, float_format="%.6f")
        print(f"Saved: {outFile}")

def Mag_guspi_upward(input_field, fish_depth, dx, dy,
                     wl, ws, zlev, pad_factor, OPT=None):
    """
    GUSPI upward continuation using Guspi (1987) method
    with cosine-tapered bandpass filter.

    Parameters
    ----------
    input_field : 2D ndarray
        Magnetic field
    fish_depth : scalar or 2D ndarray
        Measurement depth (positive downward)
    dx, dy : float
        Grid spacing
    wl : float
        Long wavelength cutoff
    ws : float
        Short wavelength cutoff
    zlev : float
        Target continuation depth
    pad_factor : float
        Symmetric padding factor
    OPT : list [max_iter, max_terms, gtol, tol], optional

    Returns
    -------
    upward_field : 2D ndarray
        Upward continued magnetic field
    """

    # -----------------------------
    # Step 1: Defaults
    if OPT is None:
        OPT = [300, 300, 0.01, 0.01]

    max_iter, max_terms, gtol, tol = OPT
    max_iter = int(max_iter)
    max_terms = int(max_terms)

    # HARD safety cap (Guspi never needs more)
    max_terms = min(max_terms, 50)

    ny, nx = input_field.shape

    # -----------------------------
    # Step 2: Symmetric padding
    pad_size = int(round(pad_factor * min(ny, nx)))

    input_padded = np.pad(input_field, pad_size, mode='reflect')

    if np.isscalar(fish_depth):
        depth_padded = np.full_like(input_padded, fish_depth)
    else:
        depth_padded = np.pad(fish_depth, pad_size, mode='reflect')

    ny_pad, nx_pad = input_padded.shape

    # -----------------------------
    # Step 3: Reference depth
    zref = np.mean(depth_padded)
    h = depth_padded - zref
    dz = zlev - zref

    # -----------------------------
    # Step 4: Mean center
    mean_input = np.mean(input_padded)
    field_centered = input_padded - mean_input

    # -----------------------------
    # Step 5: Wavelength limits
    if ws == 0:
        ws = max(dx, dy)
    if wl == 0:
        wl = min(nx * dx, ny * dy)
    if ws >= wl:
        raise ValueError("Short wavelength cutoff must be < long wavelength cutoff")

    # -----------------------------
    # Step 6: Wavenumber grid (MATLAB-faithful)
    kx = ifftshift(
        np.arange(-nx_pad // 2, nx_pad // 2) * (2 * np.pi / (nx_pad * dx))
    )
    ky = ifftshift(
        np.arange(-ny_pad // 2, ny_pad // 2) * (2 * np.pi / (ny_pad * dy))
    )
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    # -----------------------------
    # Step 7: Cosine-tapered bandpass filter
    f_low = 1.0 / wl
    f_high = 1.0 / ws

    k_low = 2 * np.pi * f_low
    k_high = 2 * np.pi * f_high

    dk = k_high - k_low
    dk_taper = 0.3 * dk

    k1 = k_low - dk_taper
    k2 = k_low
    k3 = k_high
    k4 = k_high + dk_taper

    W = np.zeros_like(K)

    idx1 = (K >= k1) & (K < k2)
    W[idx1] = 0.5 * (1 + np.cos(np.pi * (K[idx1] - k2) / dk_taper))

    idx2 = (K >= k2) & (K <= k3)
    W[idx2] = 1.0

    idx3 = (K > k3) & (K <= k4)
    W[idx3] = 0.5 * (1 + np.cos(np.pi * (K[idx3] - k3) / dk_taper))

    # -----------------------------
    # Step 8: Guspi iteration
    m_ref = field_centered.copy()
    best_m_ref = m_ref.copy()
    best_rms_err = np.inf
    bad_iter_count = 0
    max_bad_iters = 3

    print("Starting Guspi upward continuation")
    print(f"  Depth range: ref {zref:.2f} → target {zlev:.2f} (dz = {dz:.2f})")
    print(f"  Wavelength band: {ws:.2f} – {wl:.2f}")

    # Safety checks
    for arr, name in [(field_centered, "field"),
                      (h, "depth"),
                      (W, "filter"),
                      (K, "wavenumber")]:
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or Inf")

    for it in range(max_iter):
        M_fft = fft2(m_ref)
        taylor_sum = np.zeros_like(m_ref)

        fact = 1.0
        h_pow = np.ones_like(h)
        K_pow = np.ones_like(K)

        for n in range(1, max_terms + 1):
            fact *= n
            h_pow *= (-h)
            K_pow *= K

            h_term = h_pow / fact
            freq_term = K_pow * M_fft * W
            spatial_term = np.real(ifft2(freq_term))
            contribution = h_term * spatial_term

            taylor_sum += contribution

            if np.max(np.abs(contribution)) < gtol:
                print(f"  Iter {it+1}: Taylor converged at term {n}")
                break

        m_new = field_centered - taylor_sum

        rms_err = np.sqrt(np.mean((m_new - m_ref) ** 2))
        print(f"  Iter {it+1:3d}: RMS error = {rms_err:.6e}")

        if not np.isfinite(m_new).all():
            print("NaN/Inf detected — aborting")
            return best_m_ref + mean_input

        if rms_err < best_rms_err:
            best_rms_err = rms_err
            best_m_ref = m_new.copy()
            bad_iter_count = 0
        else:
            bad_iter_count += 1
            print(f"  RMS increased (bad step {bad_iter_count}/{max_bad_iters})")
            if bad_iter_count >= max_bad_iters:
                print("Too many bad steps — reverting to best result")
                return best_m_ref + mean_input

        if rms_err < tol:
            print(f"Converged at iteration {it+1}")
            break

        m_ref = m_new

    # -----------------------------
    # Step 9: Final upward continuation
    upward_filter = np.exp(-K * dz)
    m_continued = np.real(ifft2(fft2(best_m_ref) * upward_filter))
    upward_padded = m_continued + mean_input

    # -----------------------------
    # Step 10: Crop padding
    upward_field = upward_padded[
        pad_size:-pad_size,
        pad_size:-pad_size
    ]

    return upward_field
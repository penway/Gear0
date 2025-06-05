# -*- coding: utf-8 -*-
"""
gear_vibration_pipeline.py

A script to perform a first-pass diagnostic analysis of multichannel gearbox
vibration data. This pipeline implements the "Mathematical-Only Outline"
discussed, using the provided system parameters.

Pipeline Steps:
1.  Load and Pre-process Data: Detrend and normalize each channel.
2.  Spectral Survey: Calculate the Power Spectral Density (PSD) to identify
    key frequency components.
3.  Narrow-Band Filtering: Isolate specific frequency bands of interest,
    particularly the gear mesh harmonics and their sidebands.
4.  Analytic Signal Analysis: Use the Hilbert transform to calculate the
    signal envelope and instantaneous phase for demodulation analysis.
5.  Cross-Channel Metrics: Calculate metrics like envelope correlation to
    understand spatial relationships.
6.  Fault-Indicative Statistics: Compute scalar features like Kurtosis and
    Sideband Level Ratio (SLR) to quantify fault conditions.

Created on: May 28, 2025
"""

import os
import numpy as np
import scipy.io as sio
from scipy.signal import welch, firwin, filtfilt, hilbert
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

# === Additional imports for mode‑aligned, cluster‑wise analysis ===
from scipy.signal import decimate
try:
    from sklearn.cluster import SpectralClustering
except ImportError:
    SpectralClustering = None
    print("Warning: scikit‑learn not found. Spectral clustering will fall back to a simple threshold method.")

# --- 0. SCRIPT PARAMETERS AND SYSTEM CONSTANTS ---
# #############################################################################
# !!! USER: PLEASE MODIFY THIS PATH !!!
DATA_DIRECTORY = "/Users/penway/Projects/Gear0/20201028/Throughput/"
# #############################################################################

# System-specific constants
FS = 256000.0         # Sampling Frequency (Hz)
GEAR_MESH_FREQ = 766.666667 # (Hz)
PINION_SHAFT_FREQ = 33.333333 # (Hz)
N_CHANNELS = 27

# Analysis parameters
# Using a long window for high frequency resolution to resolve sidebands
WINDOW_LEN = 2**16  # 65536 samples
WINDOW_OVERLAP = 0.5 # 50% overlap for Welch's method
FFT_LEN = WINDOW_LEN # Using window length for FFT length is common

# Filter design parameters for the first gear mesh harmonic
GMF1_CENTER_FREQ = GEAR_MESH_FREQ
# Bandwidth should capture at least the first few sidebands (e.g., +/- 3*f_shaft)
# Original: GMF1_BANDWIDTH = 2.5 * PINION_SHAFT_FREQ * 2 # ~166 Hz
# Let's make it slightly wider to ensure sidebands are well within the passband
GMF1_BANDWIDTH = 3.5 * PINION_SHAFT_FREQ * 2 # ~233 Hz
FILTER_ORDER = 4096 # Filter order (taps); ensure it's even for numtaps=order+1
                    # For firwin, numtaps is the number of coefficients.
                    # If order is N, numtaps is N+1.


def load_data(directory, num_channels):
    """
    Loads the 27 channel .mat files into a single NumPy array.

    Args:
        directory (str): Path to the folder containing the data files.
        num_channels (int): The number of channels to load (e.g., 27).

    Returns:
        np.ndarray: A (num_channels, N_samples) array of raw data.
                    Returns None if the directory is not found.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        print("Please update the DATA_DIRECTORY variable in the script.")
        return None

    data_list = []
    print("Loading data...")
    for i in range(1, num_channels + 1):
        filename = f"3027_217_Data_Chan{i}.mat"
        filepath = os.path.join(directory, filename)
        try:
            datai_mat = sio.loadmat(filepath)
            # Assuming 'Data1' is the key for the actual time series data
            datai_arr = np.array(datai_mat['Data1'])
            data_list.append(datai_arr)
        except FileNotFoundError:
            print(f"Error: Could not find file {filepath}")
            return None
        except KeyError:
            print(f"Error: Key 'Data1' not found in {filepath}. Available keys: {datai_mat.keys()}")
            return None


    # Stack into a single numpy array and remove the singleton dimension
    try:
        data = np.array(data_list)
        data = np.squeeze(data, axis=1) # From (27, 1, N) to (27, N)
        print(f"Data loaded successfully. Shape: {data.shape}")
    except ValueError as e:
        print(f"Error processing loaded data. Data list shapes might be inconsistent: {e}")
        for idx, arr in enumerate(data_list):
            print(f"Shape of data from Chan{idx+1}.mat: {arr.shape}")
        return None
    return data

def step_1_preprocess(X):
    """
    Performs Step 1: Detrending & Normalization.
    Removes DC offset and scales each channel to unit variance.

    Args:
        X (np.ndarray): The raw data matrix (channels x samples).

    Returns:
        np.ndarray: The normalized data matrix X_tilde.
    """
    print("Step 1: Pre-processing (Detrend & Normalize)...")
    # mu_i = E[x_i]
    mu = np.mean(X, axis=1, keepdims=True)
    # sigma_i^2 = E[(x_i - mu_i)^2]
    sigma = np.std(X, axis=1, keepdims=True)

    # Handle cases where a channel might be flat (sigma = 0) to avoid NaN/Inf
    sigma[sigma == 0] = 1.0 # Replace 0 std with 1 to avoid division by zero

    X_tilde = (X - mu) / sigma
    print("Pre-processing complete.")
    return X_tilde

def step_2_spectral_survey(X_tilde, fs, window_len, overlap_frac, fft_len):
    """
    Performs Step 2: Spectral Survey using Welch's method.

    Args:
        X_tilde (np.ndarray): The pre-processed data (channels x samples).
        fs (float): Sampling frequency.
        window_len (int): Length of the window for FFT.
        overlap_frac (float): Fraction of overlap between windows.
        fft_len (int): Length of the FFT.

    Returns:
        tuple: (frequencies, psd_matrix)
               - frequencies (np.ndarray): Array of frequency bins.
               - psd_matrix (np.ndarray): PSD for each channel.
    """
    print("Step 2: Performing Spectral Survey (Welch's PSD)...")
    noverlap = int(window_len * overlap_frac)
    frequencies, psd_matrix = welch(
        X_tilde,
        fs=fs,
        window='hann',
        nperseg=window_len,
        noverlap=noverlap,
        nfft=fft_len, # Specify FFT length
        axis=1, # perform along the time axis
        average='mean',
        detrend=False # Data is already detrended (mean removed)
    )
    print("Spectral survey complete.")
    return frequencies, psd_matrix

def step_3_narrowband_filter(X_tilde, fs, center_freq, bandwidth, order):
    """
    Performs Step 3: Narrow-Band Extraction using an FIR filter.
    Note: Mathematical outline had this as step 4, but logically it's better here.

    Args:
        X_tilde (np.ndarray): The pre-processed data.
        fs (float): Sampling frequency.
        center_freq (float): Center frequency of the band to extract.
        bandwidth (float): Width of the band to extract.
        order (int): The order of the FIR filter. numtaps = order + 1.

    Returns:
        np.ndarray: The filtered signal y_k for all channels.
    """
    print(f"Step 3: Designing and applying FIR filter at {center_freq:.2f} Hz...")
    nyquist = fs / 2.0
    # Define the cutoff frequencies for the bandpass filter
    low_cut = center_freq - (bandwidth / 2.0)
    high_cut = center_freq + (bandwidth / 2.0)

    # Ensure cutoffs are valid
    if low_cut <= 0:
        low_cut = 1e-3 # Small positive number to avoid issues with 0 Hz
        print(f"Warning: Low cut-off was <= 0. Adjusted to {low_cut:.2e} Hz.")
    if high_cut >= nyquist:
        high_cut = nyquist - 1e-3 # Slightly less than Nyquist
        print(f"Warning: High cut-off was >= Nyquist. Adjusted to {high_cut:.2e} Hz.")
    if low_cut >= high_cut:
        print(f"Error: Low cut-off ({low_cut}) >= high cut-off ({high_cut}). Adjust filter parameters.")
        # Return original signal or handle error appropriately
        return X_tilde # Or raise an error

    # Design the FIR filter using a Hann window
    # numtaps is the filter order + 1. It should be odd for Type I (symmetric) BP filter.
    # If 'order' is meant as the number of taps, then use 'order'.
    # If 'order' is the mathematical order N, then numtaps = N+1.
    # Let's assume 'order' is N, so numtaps = order + 1.
    # firwin expects numtaps. If filter_order is N, then numtaps = N+1.
    # For a Type I filter (symmetric, odd length), numtaps should be odd.
    # So, if 'order' is even, order+1 is odd. This is good.
    num_taps = order + 1
    if num_taps % 2 == 0: # Ensure num_taps is odd for symmetric filter
        num_taps += 1
        print(f"Adjusted filter num_taps to {num_taps} to be odd for Type I filter.")


    fir_coeffs = firwin(
        numtaps=num_taps,
        cutoff=[low_cut, high_cut],
        pass_zero=False, # This makes it a bandpass filter
        fs=fs,
        window='hann'
    )

    # Apply the filter to all channels using filtfilt for zero phase delay
    y_k = filtfilt(fir_coeffs, 1.0, X_tilde, axis=1)
    print("Filtering complete.")
    return y_k

def step_4_analytic_signal(y_k):
    """
    Performs Step 4: Analytic Signal & Instantaneous Quantities.
    (Corresponds to Step 5 in mathematical outline)

    Args:
        y_k (np.ndarray): The narrow-band filtered signal.

    Returns:
        tuple: (envelope, unwrapped_phase)
               - envelope (np.ndarray): The signal envelope (a_k).
               - unwrapped_phase (np.ndarray): The unwrapped inst. phase (phi_k).
    """
    print("Step 4: Calculating Analytic Signal (Hilbert Transform)...")
    # z_k(n) = y_k(n) + j*H{y_k(n)}
    z_k = hilbert(y_k, axis=1)

    # a_k(n) = |z_k(n)|
    a_k = np.abs(z_k)

    # phi_k(n) = arg(z_k(n))
    phi_k = np.unwrap(np.angle(z_k), axis=1)
    print("Analytic signal calculation complete.")
    return a_k, phi_k

def step_5_envelope_correlation(a_k):
    """
    Performs Step 5: Cross-Channel Metrics - Envelope Correlation.
    (Corresponds to part of Step 6 in mathematical outline)

    Args:
        a_k (np.ndarray): The envelope signals for all channels.

    Returns:
        np.ndarray: The (n_channels x n_channels) correlation matrix C_k.
    """
    print("Step 5: Calculating Envelope Correlation Matrix...")
    # np.corrcoef expects rows to be variables, columns to be observations.
    # a_k is (channels, samples), which is correct.
    C_k = np.corrcoef(a_k)
    print("Correlation matrix calculation complete.")
    return C_k

def step_6_fault_statistics(y_k, a_k, freqs, psd_of_X_tilde, center_freq, shaft_freq, fs):
    """
    Performs Step 6: Fault-Indicative Statistics.
    (Corresponds to Step 8 in mathematical outline)

    Args:
        y_k (np.ndarray): The filtered signal for kurtosis calculation.
        a_k (np.ndarray): The envelope signal for envelope spectrum analysis.
        freqs (np.ndarray): Frequency bins from the PSD of X_tilde.
        psd_of_X_tilde (np.ndarray): PSD matrix of X_tilde (channels x freqs).
        center_freq (float): The carrier frequency (e.g., GMF).
        shaft_freq (float): The modulating frequency (e.g., shaft speed).
        fs (float): Sampling frequency (for envelope spectrum).

    Returns:
        tuple: (kurt_val, slr, env_spec_freqs, env_spec)
               - kurt_val (np.ndarray): Kurtosis for each channel's filtered signal.
               - slr (np.ndarray): Sideband Level Ratio for each channel from X_tilde PSD.
               - env_spec_freqs (np.ndarray): Frequencies for envelope spectrum.
               - env_spec (np.ndarray): Envelope spectrum for each channel.
    """
    print("Step 6: Calculating Fault-Indicative Statistics...")
    # Kurtosis on the filtered time-series data
    # Fisher=False gives Pearson's kurtosis (3 for Gaussian)
    kurt_val = kurtosis(y_k, axis=1, fisher=False)

    # Sideband Level Ratio (SLR) from the PSD of X_tilde
    # Find the indices for the carrier and sideband frequencies in freqs
    carrier_idx = np.argmin(np.abs(freqs - center_freq))
    # Define a small band around the target frequencies to sum power
    # This makes it more robust to slight frequency shifts or FFT binning
    freq_resolution = freqs[1] - freqs[0]
    integration_bins = max(1, int( (shaft_freq / 4) / freq_resolution) ) # Integrate over ~1/4 of shaft freq bandwidth

    def get_power_in_band(psd_data, target_freq_idx, num_bins_around):
        start_idx = max(0, target_freq_idx - num_bins_around)
        end_idx = min(psd_data.shape[1]-1, target_freq_idx + num_bins_around)
        return np.sum(psd_data[:, start_idx : end_idx+1], axis=1) * freq_resolution


    carrier_power = get_power_in_band(psd_of_X_tilde, carrier_idx, integration_bins)

    sb1_minus_idx = np.argmin(np.abs(freqs - (center_freq - shaft_freq)))
    sb1_plus_idx = np.argmin(np.abs(freqs - (center_freq + shaft_freq)))

    sideband_power_minus = get_power_in_band(psd_of_X_tilde, sb1_minus_idx, integration_bins)
    sideband_power_plus = get_power_in_band(psd_of_X_tilde, sb1_plus_idx, integration_bins)
    total_sideband_power = sideband_power_minus + sideband_power_plus


    # Handle potential division by zero or log of zero
    valid_mask = (carrier_power > 1e-9) & (total_sideband_power > 1e-9) # Check both are non-zero
    slr = np.full(psd_of_X_tilde.shape[0], -100.0) # Default for invalid cases (very low dB)

    slr[valid_mask] = 10 * np.log10(total_sideband_power[valid_mask] / carrier_power[valid_mask])
    slr[~valid_mask & (carrier_power <= 1e-9)] = 0 # If carrier is zero, but sidebands exist, could be high positive
    slr[~valid_mask & (total_sideband_power <= 1e-9) & (carrier_power > 1e-9)] = -100.0 # If sidebands are zero, very low dB

    # Envelope Spectrum (for one channel, e.g., channel 0)
    # Detrend envelope before FFT (remove DC)
    a_k_detrended = a_k - np.mean(a_k, axis=1, keepdims=True)
    
    # Using Welch for smoother envelope spectrum
    # Window length for envelope spectrum should be long enough to resolve shaft_freq
    # For example, if shaft_freq is 33Hz, we need resolution better than that.
    # T_window = 1 / desired_freq_res. L_window = T_window * fs_envelope
    # The envelope signal a_k has the same "sampling rate" as the original signal.
    env_window_len = WINDOW_LEN # Can use the same long window
    env_noverlap = int(env_window_len * WINDOW_OVERLAP)

    env_spec_freqs, env_spec = welch(
        a_k_detrended, # Use detrended envelope
        fs=fs, # Envelope signal has same effective sampling rate
        window='hann',
        nperseg=env_window_len,
        noverlap=env_noverlap,
        nfft=env_window_len, # FFT length
        axis=1,
        average='mean',
        detrend=False # Already detrended
    )

    print("Fault statistics calculation complete.")
    return kurt_val, slr, env_spec_freqs, env_spec


# ------------------------------------------------------------------
# MODE‑ALIGNED, CLUSTER‑WISE ANALYSIS HELPERS
# ------------------------------------------------------------------
def compute_plv(phase_matrix):
    """
    Compute pair‑wise Phase‑Locking Value (PLV).

    Args
    ----
    phase_matrix : ndarray (n_channels, n_samples)
        Unwrapped instantaneous phase of analytic signals.

    Returns
    -------
    plv : ndarray (n_channels, n_channels)
    """
    exp_phase = np.exp(1j * phase_matrix)
    plv = np.abs(exp_phase @ exp_phase.conj().T) / phase_matrix.shape[1]
    return plv


def cluster_sensors_with_plv(plv_matrix, n_clusters=3, threshold=0.55):
    """
    Cluster sensors either via SpectralClustering (if sklearn available) or
    via a simple similarity threshold.

    Returns
    -------
    clusters : list[list[int]]
        List of channel‑index lists (0‑based).
    labels   : ndarray (n_channels,)  (cluster id per channel)
    """
    n_ch = plv_matrix.shape[0]

    if SpectralClustering is not None:
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=0
        )
        labels = model.fit_predict(plv_matrix)
    else:
        # Fallback: simple grouping by threshold on average PLV
        labels = -np.ones(n_ch, dtype=int)
        current_label = 0
        for i in range(n_ch):
            if labels[i] != -1:
                continue
            labels[i] = current_label
            for j in range(i + 1, n_ch):
                if plv_matrix[i, j] >= threshold:
                    labels[j] = current_label
            current_label += 1

    clusters = [np.where(labels == k)[0].tolist()
                for k in range(labels.max() + 1)]
    return clusters, labels


def phase_align_and_average_envelope(envelopes, clusters):
    """
    Time‑shift envelopes within each cluster to maximise cross‑correlation
    w.r.t. the first sensor in the cluster, then average.

    Args
    ----
    envelopes : ndarray (n_channels, n_samples)
    clusters  : list of list of int

    Returns
    -------
    avg_envs : list[np.ndarray]  (per cluster averaged envelope)
    """
    avg_envs = []
    for cl in clusters:
        if len(cl) == 0:
            continue
        ref_ch = cl[0]
        ref_env = envelopes[ref_ch]
        aligned_envs = []
        for ch in cl:
            env = envelopes[ch]
            # FFT cross‑correlation for speed
            corr = np.fft.ifft(
                np.fft.fft(ref_env) * np.conj(np.fft.fft(env))
            ).real
            lag = np.argmax(corr)
            if lag > len(env) // 2:
                lag -= len(env)
            env_shifted = np.roll(env, -lag)
            aligned_envs.append(env_shifted)
        avg_envs.append(np.mean(np.vstack(aligned_envs), axis=0))
    return avg_envs


def decimate_signal(signal, fs_orig, shaft_freq, nyq_frac=0.25):
    """
    Decimate so that shaft frequency is ~nyq_frac of new Nyquist.

    Returns
    -------
    sig_dec : ndarray
    fs_new  : float
    """
    target_nyq = shaft_freq / nyq_frac
    dec_factor = int(round(fs_orig / (2 * target_nyq)))
    if dec_factor < 1:
        dec_factor = 1
    sig_dec = decimate(signal, dec_factor, ftype='fir', zero_phase=True)
    fs_new = fs_orig / dec_factor
    return sig_dec, fs_new


def visualize_results(raw_data, X_tilde, freqs, psd_matrix, y_k, a_k, C_k, kurt_val, slr,
                      env_spec_freqs, env_spec_all_channels):
    """
    Generates plots to visualize the results from each pipeline step.
    Args:
        X_tilde: Added this missing parameter
        (other parameters as before)
        env_spec_freqs: Frequencies for envelope spectrum
        env_spec_all_channels: Envelope spectrum for all channels
    """
    print("Generating visualizations...")
    # Attempt to use a more modern style if available, otherwise default
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        print("seaborn-v0_8-whitegrid style not found, using default.")
        # No specific fallback needed, matplotlib will use its default

    # Choose a sample channel to plot (e.g., channel 0)
    ch = 0
    time_vector = np.arange(raw_data.shape[1]) / FS

    # --- Figure 1: Spectral Survey (PSD) & Envelope Spectrum ---
    fig = plt.figure(figsize=(18, 12)) # Increased figure size
    plt.suptitle("Gearbox Vibration Analysis Pipeline Results", fontsize=20, y=0.99)

    # --- Ax1: Average PSD of X_tilde ---
    ax1 = fig.add_subplot(2, 2, 1)
    avg_psd_X_tilde = np.mean(psd_matrix, axis=0) # psd_matrix is from X_tilde
    ax1.semilogy(freqs, avg_psd_X_tilde)
    ax1.set_title(f"Step 2: Avg. PSD of Normalized Data (X_tilde)\n(L={WINDOW_LEN}, Hann Window)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power/Frequency (dB/Hz or V^2/Hz)")
    ax1.set_xlim(0, 5 * GEAR_MESH_FREQ) # Zoom into relevant frequency range
    ax1.set_ylim(bottom=max(1e-9, np.min(avg_psd_X_tilde[freqs < 5 * GEAR_MESH_FREQ])/10) ) # Avoid zero or too low limits for log scale

    # Mark gear mesh harmonics and potential sidebands
    for i in range(1, 6): # Up to 5th harmonic
        ax1.axvline(x=i * GEAR_MESH_FREQ, color='r', linestyle='--', alpha=0.7,
                    label=f'{i}x GMF' if i == 1 else None)
        # Indicate sideband regions
        ax1.axvline(x=i * GEAR_MESH_FREQ - PINION_SHAFT_FREQ, color='g', linestyle=':', alpha=0.5,
                    label='Sidebands (GMF ± Shaft)' if i == 1 else None)
        ax1.axvline(x=i * GEAR_MESH_FREQ + PINION_SHAFT_FREQ, color='g', linestyle=':', alpha=0.5)
    ax1.legend(fontsize='small')
    ax1.grid(True, which='both', linestyle='-', alpha=0.5)


    # --- Ax2: Filtered Signal and Envelope ---
    ax2 = fig.add_subplot(2, 2, 2)
    # Plot a small time segment to see details. Choose a segment with some activity.
    # For instance, 0.1s to 0.12s (20ms duration)
    start_time_plot = 0.1
    end_time_plot = start_time_plot + 0.02 # 20 ms segment
    plot_indices = (time_vector >= start_time_plot) & (time_vector <= end_time_plot)

    # Check if plot_indices has any True values
    if np.any(plot_indices):
        ax2.plot(time_vector[plot_indices], X_tilde[ch, plot_indices], label='Normalized Raw (X_tilde)', alpha=0.6, color='gray')
        ax2.plot(time_vector[plot_indices], y_k[ch, plot_indices], label=f'Filtered (Band: {GMF1_CENTER_FREQ:.0f} Hz)', color='purple', linewidth=1.5)
        ax2.plot(time_vector[plot_indices], a_k[ch, plot_indices], label='Envelope (a_k)', color='orange', linewidth=2)
        ax2.plot(time_vector[plot_indices], -a_k[ch, plot_indices], color='orange', linewidth=2, alpha=0.7) # Negative envelope
    else:
        ax2.text(0.5, 0.5, "No data in selected plot range.", ha='center', va='center')


    ax2.set_title(f"Steps 3 & 4: Filtered Signal & Envelope (Channel {ch+1})")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude (Normalized)")
    ax2.legend(fontsize='small')
    ax2.grid(True, linestyle='-', alpha=0.5)


    # --- Ax3: Envelope Correlation Matrix ---
    ax3 = fig.add_subplot(2, 2, 3)
    im = ax3.imshow(C_k, cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
    ax3.set_title(f"Step 5: Envelope Correlation Matrix (C_k)\n(Band: {GMF1_CENTER_FREQ:.0f} Hz)")
    ax3.set_xlabel("Channel Index")
    ax3.set_ylabel("Channel Index")
    # Set ticks to show channel numbers 1-27
    tick_locs = np.arange(N_CHANNELS)
    tick_labels = [str(i+1) for i in tick_locs]
    ax3.set_xticks(tick_locs[::N_CHANNELS//5]) # Show fewer ticks if too crowded
    ax3.set_xticklabels(tick_labels[::N_CHANNELS//5])
    ax3.set_yticks(tick_locs[::N_CHANNELS//5])
    ax3.set_yticklabels(tick_labels[::N_CHANNELS//5])

    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label='Correlation Coeff.')


    # --- Ax4: Fault Indicators (Kurtosis and SLR) ---
    ax4 = fig.add_subplot(2, 2, 4)
    x_indices = np.arange(N_CHANNELS)
    bar_width = 0.35

    # Kurtosis bars
    rects1 = ax4.bar(x_indices - bar_width/2, kurt_val, bar_width, label='Kurtosis (Filtered Signal)', color='dodgerblue')
    ax4.set_ylabel("Kurtosis (Pearson's)", color='dodgerblue')
    ax4.tick_params(axis='y', labelcolor='dodgerblue')
    ax4.set_xlabel("Channel Index (1-27)")
    ax4.set_xticks(x_indices)
    ax4.set_xticklabels([str(i+1) for i in x_indices]) # Labels 1 to 27

    # SLR bars on a secondary y-axis
    ax4b = ax4.twinx()
    rects2 = ax4b.bar(x_indices + bar_width/2, slr, bar_width, label='SLR (X_tilde PSD)', color='darkorange', alpha=0.7)
    ax4b.set_ylabel("Sideband Level Ratio (dB)", color='darkorange')
    ax4b.tick_params(axis='y', labelcolor='darkorange')

    ax4.set_title("Step 6: Fault Indicative Statistics per Channel")
    # Add legends manually to avoid overlap if using fig.legend
    lines, labels = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4b.legend(lines + lines2, labels + labels2, loc='upper left', fontsize='small')
    ax4.grid(False) # Turn off grid for the primary axis if twin axes are used
    ax4b.grid(False)


    # --- Figure 2: Envelope Spectrum ---
    plt.figure(figsize=(12, 6)) # New figure for envelope spectrum
    # Plot envelope spectrum for the sample channel 'ch'
    env_spec_ch = env_spec_all_channels[ch, :]
    plt.plot(env_spec_freqs, 10 * np.log10(env_spec_ch + 1e-12)) # Plot in dB, add epsilon for log
    plt.title(f"Envelope Spectrum (Channel {ch+1} - Band: {GMF1_CENTER_FREQ:.0f} Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB re: V^2/Hz or unit^2/Hz)")
    plt.xlim(0, 5 * PINION_SHAFT_FREQ) # Focus on low frequencies where shaft speed harmonics appear
    plt.ylim(bottom=np.percentile(10*np.log10(env_spec_ch + 1e-12), 5)-10) # Dynamic Y limit

    # Mark pinion shaft frequency and its harmonics
    for i in range(1, 6): # Up to 5th harmonic of shaft speed
        plt.axvline(x=i * PINION_SHAFT_FREQ, color='m', linestyle='--', alpha=0.7,
                    label=f'{i}x Shaft Freq' if i == 1 else None)
    plt.legend(fontsize='small')
    plt.grid(True, which='both', linestyle='-', alpha=0.5)
    plt.tight_layout()


    plt.show()


def main():
    """Main execution block."""
    # --- Load Data ---
    X = load_data(DATA_DIRECTORY, N_CHANNELS)
    if X is None:
        print("Exiting due to data loading failure.")
        return # Stop execution if data loading failed

    # --- Run Pipeline ---
    # Step 1: Detrending & Normalization
    X_tilde = step_1_preprocess(X)

    # Step 2: Spectral Survey (on X_tilde)
    # Use FFT_LEN for nfft argument in welch
    freqs, psd_matrix_X_tilde = step_2_spectral_survey(X_tilde, FS, WINDOW_LEN, WINDOW_OVERLAP, FFT_LEN)

    # Step 3: Narrow-Band Extraction (from X_tilde)
    y_k = step_3_narrowband_filter(X_tilde, FS, GMF1_CENTER_FREQ, GMF1_BANDWIDTH, FILTER_ORDER)

    # Step 4: Analytic Signal (from y_k)
    a_k, phi_k = step_4_analytic_signal(y_k)

    # --- Cluster‑wise analysis on the narrow‑band mode ---
    # Compute PLV matrix from instantaneous phases
    plv_matrix = compute_plv(phi_k)
    # Choose number of clusters heuristically (3). Adjust as needed.
    clusters, labels = cluster_sensors_with_plv(plv_matrix, n_clusters=3)

    print("\n--- CLUSTER INFORMATION (based on PLV) ---")
    for idx, cl in enumerate(clusters):
        print(f"Cluster {idx}: channels { [c+1 for c in cl] }")

    # Align and average envelopes inside each cluster
    avg_envs = phase_align_and_average_envelope(a_k, clusters)

    # Decimate each averaged envelope and compute its spectrum
    cluster_env_specs = []
    cluster_env_freqs = None
    for env in avg_envs:
        env_dec, fs_dec = decimate_signal(env, FS, PINION_SHAFT_FREQ)
        # Welch on decimated envelope
        f_env, Pxx_env = welch(
            env_dec - np.mean(env_dec),
            fs=fs_dec,
            nperseg=min(len(env_dec), 16384),
            window='hann',
            noverlap=0,
            detrend=False
        )
        cluster_env_specs.append(10 * np.log10(Pxx_env + 1e-12))
        if cluster_env_freqs is None:
            cluster_env_freqs = f_env

    # Print basic discrete modulation info
    print("\nEnvelope spectra (dB) at shaft harmonics for first cluster:")
    if len(cluster_env_specs):
        spec0 = cluster_env_specs[0]
        for k in range(1, 6):
            idx = np.argmin(np.abs(cluster_env_freqs - k * PINION_SHAFT_FREQ))
            print(f"  {k}×shaft ({k*PINION_SHAFT_FREQ:.1f} Hz): {spec0[idx]:.1f} dB")

    # Visualise envelope spectra of clusters
    if len(cluster_env_specs) > 0:
        plt.figure(figsize=(10, 6))
        for i, spec in enumerate(cluster_env_specs):
            plt.plot(
                cluster_env_freqs, spec,
                label=f'Cluster {i}'
            )
        plt.title('Decimated Envelope Spectra per Cluster')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.xlim(0, 6 * PINION_SHAFT_FREQ)
        for k in range(1, 6):
            plt.axvline(k * PINION_SHAFT_FREQ, color='m', ls='--', alpha=0.6)
        plt.legend()
        plt.grid(True, which='both', ls=':')
        plt.tight_layout()
        plt.show()

    # Step 5: Cross-Channel Metrics (from a_k)
    C_k = step_5_envelope_correlation(a_k)

    # Step 6: Fault-Indicative Statistics
    # Pass psd_matrix_X_tilde for SLR calculation
    kurt_val, slr, env_spec_freqs, env_spec_all_channels = step_6_fault_statistics(
        y_k, a_k, freqs, psd_matrix_X_tilde, GMF1_CENTER_FREQ, PINION_SHAFT_FREQ, FS
    )

    # --- Print sample results to console ---
    print("\n--- SAMPLE RESULTS (First 5 channels) ---")
    print(f"Shape of raw data (X): {X.shape}")
    print(f"Shape of normalized data (X_tilde): {X_tilde.shape}")
    print(f"Shape of filtered data (y_k): {y_k.shape}")
    print(f"Shape of envelope data (a_k): {a_k.shape}")
    print(f"Shape of phase data (phi_k): {phi_k.shape}")
    print(f"Shape of correlation matrix (C_k): {C_k.shape}")
    print("\nKurtosis (Pearson's) of filtered signal (y_k):")
    print(np.round(kurt_val[:5], 2))
    print("\nSideband Level Ratio (dB) from X_tilde PSD:")
    print(np.round(slr[:5], 2))
    print(f"\nEnvelope Spectrum Frequencies range from {env_spec_freqs[0]:.2f} Hz to {env_spec_freqs[-1]:.2f} Hz")
    print(f"Shape of envelope spectrum data: {env_spec_all_channels.shape}")


    # --- Visualize ---
    # Pass X_tilde to the visualization function
    visualize_results(X, X_tilde, freqs, psd_matrix_X_tilde, y_k, a_k, C_k, kurt_val, slr,
                      env_spec_freqs, env_spec_all_channels)


if __name__ == '__main__':
    main()

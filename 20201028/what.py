# -*- coding: utf-8 -*-
"""
gear_vibration_pipeline_vmd.py

Extended gearbox vibration analysis pipeline with:
- Global spectral & envelope analysis
- Cluster-wise PLV alignment & decimation
- Single-channel VMD (via vmdpy) per channel + delay-aware shared-mode extraction

Created on: May 28, 2025
"""

import os
import numpy as np
import scipy.io as sio
from scipy.signal import welch, firwin, filtfilt, hilbert, decimate
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

# Optional imports
try:
    from sklearn.cluster import SpectralClustering
except ImportError:
    SpectralClustering = None
    print("Warning: scikit-learn not installed; clustering will fall back to thresholding.")

try:
    from vmdpy import VMD
except ImportError:
    VMD = None
    print("Warning: vmdpy not installed; VMD-based mode extraction disabled.")

# --- Constants ---
DATA_DIRECTORY = "/Users/penway/Projects/Gear0/20201028/Throughput/"
FS = 256000.0
GEAR_MESH_FREQ = 766.666667
PINION_SHAFT_FREQ = 33.333333
N_CHANNELS = 27
WINDOW_LEN = 2**16
WINDOW_OVERLAP = 0.5
FFT_LEN = WINDOW_LEN
GMF1_CENTER_FREQ = GEAR_MESH_FREQ
GMF1_BANDWIDTH = 3.5 * PINION_SHAFT_FREQ * 2
FILTER_ORDER = 4096

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def load_data(directory, num_channels):
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return None
    data_list = []
    for i in range(1, num_channels+1):
        fn = f"3027_217_Data_Chan{i}.mat"
        path = os.path.join(directory, fn)
        try:
            mat = sio.loadmat(path)
            arr = np.squeeze(mat['Data1'])
            data_list.append(arr)
        except Exception as e:
            print(f"Failed loading {fn}: {e}")
            return None
    data = np.vstack(data_list)
    print(f"Data loaded. Shape: {data.shape}")
    return data


def preprocess(X):
    mu = np.mean(X, axis=1, keepdims=True)
    sigma = np.std(X, axis=1, keepdims=True)
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma


def spectral_survey(X, fs, wl, ov, nfft):
    noverlap = int(wl * ov)
    f, Pxx = welch(X, fs=fs, window='hann', nperseg=wl,
                   noverlap=noverlap, nfft=nfft, axis=1,
                   average='mean', detrend=False)
    return f, Pxx


def narrowband_filter(X, fs, cf, bw, order):
    nyq = fs/2.0
    low = max(cf - bw/2.0, 1e-3)
    high = min(cf + bw/2.0, nyq - 1e-3)
    taps = order+1
    if taps % 2 == 0:
        taps += 1
    coeffs = firwin(taps, [low, high], pass_zero=False, fs=fs, window='hann')
    return filtfilt(coeffs, 1.0, X, axis=1)


def analytic_signal(Y):
    Z = hilbert(Y, axis=1)
    return np.abs(Z), np.unwrap(np.angle(Z), axis=1)


def envelope_correlation(A):
    return np.corrcoef(A)


def fault_statistics(Y, A, freqs, psd, cf, sf, fs):
    kurt = kurtosis(Y, axis=1, fisher=False)
    df = freqs[1] - freqs[0]
    idx_c = np.argmin(np.abs(freqs - cf))
    idx_minus = np.argmin(np.abs(freqs - (cf - sf)))
    idx_plus = np.argmin(np.abs(freqs - (cf + sf)))
    get_pow = lambda idx: np.sum(psd[:, idx-1:idx+2], axis=1) * df
    p_car = get_pow(idx_c)
    p_sb = get_pow(idx_minus) + get_pow(idx_plus)
    slr = 10 * np.log10(np.where(p_car>0, p_sb/p_car, 1e-12))
    A0 = A - np.mean(A, axis=1, keepdims=True)
    f_env, P_env = welch(A0, fs=fs, window='hann', nperseg=WINDOW_LEN,
                         noverlap=int(WINDOW_LEN*WINDOW_OVERLAP), nfft=WINDOW_LEN,
                         axis=1, average='mean', detrend=False)
    return kurt, slr, f_env, P_env

# ------------------------------------------------------------------
# CLUSTER-WISE HELPERS
# ------------------------------------------------------------------

def compute_plv(phi):
    EP = np.exp(1j * phi)
    return np.abs(EP @ EP.conj().T) / phi.shape[1]


def cluster_sensors(sim, n_clusters=3, thresh=0.55):
    n = sim.shape[0]
    if SpectralClustering:
        model = SpectralClustering(n_clusters=n_clusters,
                                   affinity='precomputed',
                                   assign_labels='kmeans', random_state=0)
        labels = model.fit_predict(sim)
    else:
        labels = -np.ones(n, dtype=int)
        lab = 0
        for i in range(n):
            if labels[i] >= 0: continue
            labels[i] = lab
            for j in range(i+1, n):
                if sim[i,j] >= thresh: labels[j] = lab
            lab += 1
    return [np.where(labels==k)[0].tolist() for k in range(labels.max()+1)]


def align_and_avg(env, clusters):
    out = []
    for cl in clusters:
        if not cl: continue
        ref = env[cl[0]]
        aligned = []
        for ch in cl:
            c = np.fft.ifft(np.fft.fft(ref) * np.conj(np.fft.fft(env[ch]))).real
            lag = np.argmax(c)
            if lag > len(ref)//2: lag -= len(ref)
            aligned.append(np.roll(env[ch], -lag))
        out.append(np.mean(aligned, axis=0))
    return out


def decimate_and_spectrum(env_avg, fs, shaft_f):
    target_nyq = shaft_f / 0.25
    D = max(int(round(fs/(2*target_nyq))), 1)
    ed = decimate(env_avg, D, ftype='fir', zero_phase=True)
    fsd = fs / D
    f, P = welch(ed - ed.mean(), fs=fsd,
                 window='hann', nperseg=min(len(ed),16384), noverlap=0)
    return f, 10*np.log10(P + 1e-12)

# ------------------------------------------------------------------
# VMD HELPERS
# ------------------------------------------------------------------

def compute_vmd_modes_all_channels(X, alpha=2000, tau=0.0, K=5, DC=False, init=1, tol=1e-6):
    if VMD is None:
        return None
    n_ch, n_s = X.shape
    modes_arr = np.zeros((K, n_ch, n_s))
    for ch in range(n_ch):
        # vmdpy expects x of length n_s
        u, u_hat, omega = VMD(X[ch], alpha, tau, K, DC, init, tol)
        # u.shape = (K, n_s)
        modes_arr[:, ch, :] = u
    return modes_arr

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    X = load_data(DATA_DIRECTORY, N_CHANNELS)
    if X is None: return

    # 1. Preprocess
    X_t = preprocess(X)

    # 2. Spectral survey
    freqs, psd = spectral_survey(X_t, FS, WINDOW_LEN, WINDOW_OVERLAP, FFT_LEN)

    # 3. Narrowband filter
    Yk = narrowband_filter(X_t, FS, GMF1_CENTER_FREQ, GMF1_BANDWIDTH, FILTER_ORDER)

    # 4. Analytic signal
    Ak, Phi = analytic_signal(Yk)

    # 5. Envelope correlation
    Ck = envelope_correlation(Ak)

    # 6. Fault stats
    kurt_val, slr, f_e, P_e = fault_statistics(Yk, Ak, freqs, psd,
                                            GMF1_CENTER_FREQ,
                                            PINION_SHAFT_FREQ, FS)

    # Cluster-wise envelope analysis
    plv = compute_plv(Phi)
    clusters = cluster_sensors(plv, n_clusters=3)
    print("\n--- Clusters by PLV ---")
    for i, cl in enumerate(clusters): print(f"Cluster {i}: {[c+1 for c in cl]}")
    env_avgs = align_and_avg(Ak, clusters)
    print("\n--- Cluster envelope spectra ---")
    for i, env_avg in enumerate(env_avgs):
        f_cl, P_cl = decimate_and_spectrum(env_avg, FS, PINION_SHAFT_FREQ)
        print(f"Cluster {i+1}:")
        for k in range(1,6):
            idx = np.argmin(np.abs(f_cl - k*PINION_SHAFT_FREQ))
            print(f"  {k}x: {P_cl[idx]:.1f} dB")

    # IMF-based shared mode analysis (fast fix)
    if EMD is not None:
        print("\n--- IMF-based shared mode analysis (fast fix) ---")
        # Fast fix: only decompose up to f_max via decimation
        f_max = 500.0
        dec_factor = int(FS / (2 * f_max))
        dec_factor = max(dec_factor, 1)
        print(f"Decimating by factor {dec_factor} -> new FS = {FS/dec_factor:.1f} Hz")
        # Decimate the normalized data
        X_ds = decimate(X_tilde, dec_factor, axis=1, ftype='fir', zero_phase=True)
        fs_ds = FS / dec_factor

        # Run EMD on the shorter, decimated signal
        imfs_arr = compute_imfs_all_channels(X_ds, max_imf=5)

        # Build delay-aware similarity for each IMF mode
        max_lag = int(fs_ds * 0.005)  # allow ±5 ms shifts
        sims = compute_delay_similarity(imfs_arr, max_lag_samples=max_lag)

        # Cluster and print results per mode
        for r, sim in sims.items():
            print(f"Mode {r+1} sim[0:5,0:5]:\n", np.round(sim[:5, :5], 2))
            cls = cluster_channels(sim, n_clusters=3)
            print(f"Mode {r+1} clusters: {[[c+1 for c in cl] for cl in cls]}\n")
    else:
        print("EMD unavailable; skipping IMF analysis.")


if __name__ == '__main__':
    main()

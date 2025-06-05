#!/usr/bin/env python3
# export_gear_data.py

import os
import numpy as np
import scipy.io as sio

# –– UPDATE THIS to your folder containing the 27 .mat files ––
DATA_DIR = "/Users/penway/Projects/Gear0/20201028/Throughput/"

# Gather and sort all channel files
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".mat")]
files.sort()  # ensure Chan1…Chan27 order

# Load each into a list
data_list = []
for fn in files:
    mat = sio.loadmat(os.path.join(DATA_DIR, fn))
    arr = np.squeeze(mat['Data1'])   # shape (N,)
    data_list.append(arr)

# Stack into a (27, N) array
X = np.vstack(data_list)
print(f"Loaded {len(data_list)} channels, each of length {X.shape[1]}")

# Save as compressed NumPy archive
out_path = "gear_data.npz"
np.savez_compressed(out_path, X=X)
print(f"Saved data to {out_path}")

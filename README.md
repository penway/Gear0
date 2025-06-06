# Gear0 Vibration Analysis

This repository contains a small gearbox vibration data set along with example scripts for first-pass fault diagnostics.  The raw data were collected from a test rig in October 2020 and are provided as 27 MATLAB `.mat` files under `20201028/Throughput`.

## Contents

- `20201028/Throughput/*.mat` – multichannel vibration measurements (`Data1` variable) for each sensor channel
- `20201028/getit.py` – main analysis script implementing a typical pipeline:
  1. load and normalise each channel
  2. compute power spectral density
  3. extract a narrow band around the gear mesh frequency
  4. obtain the analytic signal and envelope
  5. compute cross-channel metrics (e.g. envelope correlation)
  6. calculate statistics such as kurtosis and sideband level ratio
- `20201028/what.py` – an extended version with additional options (clustering and VMD)
- `20201028/output.py` – utility to consolidate the `.mat` files into a compressed `gear_data.npz` archive

## Quick Start

1. Install the required Python packages (tested with Python 3.9):

```bash
pip install numpy scipy matplotlib
# Optional for clustering/VMD support
pip install scikit-learn vmdpy
```

2. Update the `DATA_DIRECTORY` variable near the top of `getit.py` so that it points to the `Throughput` folder containing the `.mat` files.

3. Run the pipeline:

```bash
python 20201028/getit.py
```

The script prints basic statistics to the console and shows several diagnostic plots, including PSD, envelope correlation and fault-indicative metrics.

## Notes

- The raw `.mat` files are large (about 2 MB each) and have not been committed in their entirety here.
- `getit.ipynb` contains exploratory code equivalent to `getit.py`.
- Example MATLAB snippets for loading the data can be found in `20201028/Throughput/untitled4.m`.


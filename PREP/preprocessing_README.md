# Preprocessing Real-World Data (Empatica E4)

This repository provides preprocessing pipelines for real-world multimodal data collected with the **Empatica E4** wearable device. It supports cleaning, segmentation, and feature extraction for downstream machine-learning models.

## Repository Structure

- **`pipeline.py`**  
  Entry point that orchestrates the complete preprocessing pipeline. It loads raw Empatica E4 data, applies modality-specific preprocessing, and extracts features.

- **`empatica_prep/`**  
  Python package containing modality-specific preprocessing logic. It is organized into the following subpackages:
  - `ppg/` – Blood Volume Pulse (BVP / PPG)
  - `temp/` – Skin temperature
  - `acc/` – Acceleration
  - `eda/` – Electrodermal activity

Each subpackage encapsulates all necessary cleaning, filtering, segmentation, and feature-extraction steps for the corresponding modality.

---

## Preprocessing Pipelines

The following sections describe the preprocessing steps for each physiological signal.

---

## Skin Temperature (`temp`)

Skin temperature data is processed using the following steps:

1. Filter outliers using accelerometer (ACC) data  
2. Apply cutoffs for unrealistic temperature values  
3. Clean the data by filtering  
4. Detect outliers using the Z-score method  
5. Calculate features  

---

## Blood Volume Pulse (BVP / PPG)

Blood Volume Pulse data is processed as follows:

1. Resample accelerometer (ACC) data  
2. Apply X-RLS motion artifact filter  
3. Determine thresholds  
4. Interpolate missing values using SciPy’s monotone cubic interpolation  
5. Clean the PPG/BVP signal:  
   1. Improve signal quality  
   2. Clean using BioSPPy with a cutoff frequency of `[0.5, 12]` Hz  
   3. Remove trend  
   4. Calculate heart rate (HR)  
   5. Optional interpolation  
6. Calculate Corrected Heart Rate (TPP)  
7. Extract HRV / R-R interval (RRI) features  
8. Split the data into task-based segments  
9. Calculate features:  
   1. Derivatives (1–3)  
   2. Morphological features:
      - Systolic peak  
      - Pulse onset  
      - Dicrotic notch  
      - Diastolic peak  
      - Min and max of first derivative  
      - Min and max points of second derivative  
   3. Time-domain features using BioSPPy  
   4. Frequency-domain features  

---

## Acceleration (`acc`)

Acceleration data is processed using the following pipeline:

1. Remove noise and non-human frequencies using either:
   - Chebyshev filter (`method="my"`), or  
   - Butterworth filter (4th order)  
2. Segment data into time windows and separate into tasks  
3. Calculate features  

---

## Electrodermal Activity (`eda`)

EDA preprocessing consists of the following steps:

0. Remove and optionally store values below zero (EDA values cannot be negative)  
1. Filter EDA signal using accelerometer (ACC) data  
2. Experiment with different filter options  
3. Smooth the signal using a 3D Hamming filter  
4. Clean the signal using custom methods:  
   1. Sanitize signal using NeuroKit  
   2. Clean EDA signal using NeuroKit with BioSPPy backend  
   3. Remove values deviating more than `3 × MAD` (Median Absolute Deviation) and replace with the median (optional interpolation)  
   4. Interpolate missing values using monotone cubic method  
5. Decompose the signal using sparseEDA  
6. Calculate EDA peaks  
7. Generate EDA epochs based on tasks (windows)  
8. Generate intervals (windows/features)  
9. Calculate features such as:  
   - `scl_mean`  
   - `rise_time`  
   - `recovery_time`  
   - `accumulative_eda`  
   - `magnitude`  
   - `skin_conductance_rate`  

---

## Environment Setup (Poetry)

This project uses **Poetry** for dependency management and virtual environment creation.

### 1. Install Poetry

If Poetry is not already installed, install it using:

```bash
curl -sSL https://install.python-poetry.org | python3 -

```
Alternatively, see the official Poetry documentation for other installation methods.

### 2. Activate the Poetry Environment

After installing the dependencies, activate the virtual environment by running:

```bash
poetry env activate

```
### 3. Run the pipeline
Change the paths in the pipeline file for saving and loading the data. Then one could call the pipeline and specificy the part:
```bash
python pipeline.py --part 1
```



## Requirements

This project is developed and tested with **Python 3.12**.

### Python Version
- Python `>=3.12,<3.13`

### Dependencies

- pandas `^2.2.3`
- ipyparallel `^8.8.0`
- matplotlib `^3.9.2`
- seaborn `^0.13.2`
- optuna `^4.0.0`
- jupyter `^1.1.1`
- xgboost `^2.1.2`
- numpy `^2.1.3`
- sktime `^0.34.0`
- neurokit2 `^0.2.10`
- mne `^1.9.0`
- PyWavelets `^1.8.0`
- pytest `^8.3.4`
- pyprep `^0.4.3`
- ipykernel `^6.29.5`
- biosppy `^2.2.2`
- cvxopt `^1.3.2`
- dotmap `^1.3.30`
- peakutils `^1.3.5`
- openpyxl `^3.1.5`

All dependencies are managed using **Poetry** and defined in the files:

- `pyproject.toml`
- `poetry.lock`
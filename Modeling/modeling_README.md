## Repository Structure

This repository contains multiple experimental pipelines for cognitive load prediction using both classical machine learning and deep learning approaches. The folder structure reflects different evaluation strategies, model types, and task formulations (regression vs. classification). Every subfolder has an model_x_trial.py file which can be used to evaluate the model.

---

### `AE/` – Autoencoders (Primary)

This folder is structured like a Python package and contains the main autoencoder implementations used in this project.

Implemented architectures:
- **LSTM Autoencoder**
- **TCN Autoencoder**

These autoencoders are used for feature extraction prior to downstream machine‑learning or deep learning models.

---

### `AFTER_AUTO/`

Contains classical machine‑learning models that operate on features extracted by the autoencoders.

Included models:
- **XGBoost**
- **Support Vector Regression (SVR)**

This folder also includes scripts for **SVM fine‑tuning**.

---

### `AFTER_AUTO_CL/`

Same structure and purpose as `AFTER_AUTO`, but adapted for **classification tasks** instead of regression.

---

### `LOSO/` – Leave‑One‑Subject‑Out (Regression)

Contains model implementations evaluated using:
- **Leave‑One‑Subject‑Out (LOSO)**
- **k‑Fold cross‑validation**

Included models:
- **XGBoost**
- **SVR**

---

### `LOSO_CL/` – Leave‑One‑Subject‑Out (Classification)

Same evaluation strategy as `LOSO`, but implemented for **classification tasks**.

Included models:
- **XGBoost**
- **Support Vector Classifier (SVC)**

---

### `LOSO_DL/` – Leave‑One‑Subject‑Out (Deep Learning, Regression)

Contains deep learning models evaluated using LOSO and k‑Fold cross‑validation.

Included models:
- **Temporal Convolutional Networks (TCN)**
- **Long Short‑Term Memory Networks (LSTM)**

---

### `LOSO_DL_CL/` – Leave‑One‑Subject‑Out (Deep Learning, Classification)

Same structure as `LOSO_DL`, but adapted for **classification tasks**.

---


### `PERSONAL/` – Personalized Models (Regression)

Contains models for **personalized cognitive load prediction**.

- Training strategy: **5‑Fold cross‑validation**
- Included models:
  - **XGBoost**
  - **SVR**

---


## Requirements

### Python Environment (Poetry)

Each subfolder in this repository contains its own **Poetry configuration files** (`pyproject.toml` and `poetry.lock`) to create an isolated and reproducible Python environment for the corresponding experiments.

This allows each modeling setup (e.g., LOSO, personalized, deep learning, classification) to be run independently with its required dependencies.

---

### Creating and Activating an Environment

To create the environment for a specific experiment, navigate to the corresponding subfolder and run:

```bash
poetry install 
```
This will create a virtual environment and install all required dependencies defined in the local Poetry files.
To activate the environment, run:
```bash
poetry env activate
```
The command will print a shell activation command (e.g., source .../activate).
Copy and execute the printed command in your terminal to activate the environment.

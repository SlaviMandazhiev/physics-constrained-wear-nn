# Physics-Constrained Neural Network for Tool Wear Prediction

A two-stage pipeline for predicting tool flank wear (VB) during CNC turning from force sensor signals.

**Stage 1** extracts a statistical feature matrix from raw force recordings.
**Stage 2** trains a physics-constrained neural network that predicts wear with monotonicity guaranteed by construction.

---

## Background

During turning, three force components (Fx, Fy, Fz) are recorded continuously at high frequency for every turning pass (*Drehen*). Tool flank wear (VB, in µm) is measured manually at sparse, irregular intervals. The challenge is to predict the full wear trajectory from the dense force signal given only a handful of ground-truth VB measurements.

The model addresses this by:
- Predicting **wear increments** (ΔVB) rather than absolute wear
- Enforcing **monotonicity architecturally** via a shifted softplus activation — wear can never decrease, without any post-processing
- Adding **phase-aware loss terms** that penalise predictions deviating from the expected curve shape in each of the three classical wear phases (break-in, steady-state, catastrophic)

---

## Repository structure

```
├── signal_processing.py                    # General-purpose signal processing library (reusable)
├── build_feature_matrix.py                 # Pipeline: raw signals → feature matrix
├── NN_PhysicsConstrained_PhaseAware_v1.py  # Physics-constrained NN training
├── requirements.txt
└── README.md
```

**`signal_processing.py`** is a standalone module — it has no dependency on the folder structure or dataset and can be imported into any project that works with force sensor time series.

---

## Pipeline overview

```
Raw force signal CSVs
        │
        ▼
[signal_processing.py]
  • Threshold segmentation
  • Edge trimming
  • Signal splitting into thirds
  • Statistical feature extraction
        │
        ▼
[build_feature_matrix.py]
  • Walks run/Drehen folder structure
  • Pairs each signal third with interpolated VB
  • Outputs feature matrix (.xlsx)
        │
        ▼
[NN_PhysicsConstrained_PhaseAware_v1.py]
  • Physics-constrained MLP training
  • Phase-aware loss terms
  • SHAP feature importance
        │
        ▼
  Predictions + metrics
```

---

## Installation

```bash
pip install -r requirements.txt
```

For GPU support, install the appropriate PyTorch build from [pytorch.org](https://pytorch.org) before running pip.

---

## Stage 1 — Signal processing and feature extraction

### `signal_processing.py` — reusable core

This module provides general-purpose utilities that work on any force signal data:

| Function | Description |
|---|---|
| `segment_active_signal()` | Extracts active cutting periods via thresholding, gap bridging, and edge trimming |
| `split_into_thirds()` | Splits a signal into 3 non-overlapping equal parts |
| `compute_features()` | Computes 9 statistical descriptors for a signal segment |
| `detect_force_columns()` | Flexibly maps Fx/Fy/Fz column names from a DataFrame |
| `read_table()` | Loads CSV or Excel files |

**Features computed per axis (27 total for Fx, Fy, Fz):**

| Feature | Description |
|---|---|
| `segment_length` | Number of samples in the window |
| `mean` | Arithmetic mean |
| `max` / `min` | Peak values |
| `std` | Standard deviation |
| `skewness` | Asymmetry of the force distribution |
| `kurtosis` | Tail heaviness (Fisher definition) |
| `area_under_curve` | Signal integral (trapezoidal rule) |
| `fft_energy` | Total spectral power (sum of squared FFT magnitudes) |

**Segmentation steps** (all parameters configurable via CLI):
1. Threshold — keep samples where `|F| > threshold` (default 150 N)
2. Gap bridging — merge gaps shorter than `min_break_length` (default 100 samples)
3. Length filter — drop segments shorter than `min_segment_length` (default 5000 samples)
4. Edge trimming — remove `trim_samples` from each boundary (default 1600 samples ≈ 1 s at 1600 Hz)
5. Concatenate surviving segments

### `build_feature_matrix.py` — dataset pipeline

Walks a specific run/Drehen folder structure, applies the signal processing, and pairs each signal third with an interpolated VB value.

**Required inputs:**

1. **Run folders** — a root directory containing one subdirectory per tool sample. Each run folder holds Drehen subfolders (`01-Drehen1`, `02-Drehen2`, ...), each with a downsampled force signal CSV.

2. **Interpolated VB file** (`.xlsx`) with columns:

| Column | Description |
|---|---|
| `Reihe` | Test series number |
| `Schichtsystem` | Coating system label |
| `Sample_ID` | Matches the run folder identifier |
| `cutting_length_m` | Cumulative cutting length in metres |
| `VB_fit_um` | Interpolated flank wear in µm |
| `is_original` | `True` for measured points, `False` for interpolated |

**Usage:**

```bash
python build_feature_matrix.py \
    --runs_root /path/to/run/folders \
    --combined_vb_xlsx /path/to/vb_interpolated.xlsx \
    --output FEATURE_MATRIX.xlsx
```

| Argument | Default | Description |
|---|---|---|
| `--threshold` | `150.0` | Force threshold in N |
| `--min_segment_length` | `5000` | Minimum samples per segment |
| `--min_break_length` | `100` | Gap bridging threshold (samples) |
| `--trim_samples` | `1600` | Edge samples removed per segment |

**Output:** Excel file with 3 rows per Drehen — metadata, cutting length, VB target, and 27 force features.

---

## Stage 2 — Physics-constrained neural network

### How the model works

```
x_t  →  MLP  →  z_t  →  shifted_softplus  →  rate_t (≥ 0)
                                                    │
                                         ΔVB_t = rate_t × Δs_t
                                                    │
                               VB(t) = VB(0) + Σ ΔVB_i   [cumsum]
```

Monotonicity is enforced **by the model architecture** — the shifted softplus ensures `rate_t ≥ 0` at all times, making the reconstructed VB curve non-decreasing without any post-processing.

### Loss function

```
L = MSE(VB_pred, VB_true)
  + δ  · MSE(ΔVB_pred, ΔVB_true)           # wear increment supervision
  + λ₁ · L₁  (Phase 1: VB ≈ a·log(s+1)+b)  # break-in shape
  + λ₂ · L₂  (Phase 2: VB ≈ a·s+b)         # steady-state shape
  + λ₃ · L₃  (Phase 3: log(VB) ≈ a·s+b)    # catastrophic shape
```

Phase losses use differentiable least squares (`torch.linalg.solve`) inside the compute graph — gradients propagate through the fit parameters back to the model weights.

### Input

The feature matrix from Stage 1, plus one column added manually before training:

**`Verschleiss_Phase`** (integer 1/2/3) — wear phase label per row:
- `1` — break-in (rapid initial wear)
- `2` — steady-state (approximately linear)
- `3` — catastrophic (accelerating toward failure)

If absent or NaN, phase losses are simply inactive.

### Usage

```bash
python NN_PhysicsConstrained_PhaseAware_v1.py \
    --data FEATURE_MATRIX.xlsx \
    --results_dir results/my_run \
    --split_by sample_id \
    --delta_lambda 0.1 \
    --phase1_lambda 0.1 \
    --phase2_lambda 0.1 \
    --phase3_lambda 0.1
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | `FINAL_FEATURE_MATRIX.xlsx` | Feature matrix path |
| `--results_dir` | *(required)* | Output directory |
| `--split_by` | `schichtsystem` | `sample_id`, `schichtsystem`, or `run_number` |
| `--epochs` | `120` | Max training epochs |
| `--batch_size` | `16` | Samples per batch (one = one full wear trajectory) |
| `--hidden` | `128` | Hidden units per layer (2 layers) |
| `--dropout` | `0.1` | Dropout rate |
| `--lr` | `5e-4` | Adam learning rate |
| `--delta_lambda` | `0.1` | Weight on ΔVB increment loss |
| `--phase1_lambda` | `0.0` | Weight on Phase 1 shape loss |
| `--phase2_lambda` | `0.0` | Weight on Phase 2 shape loss |
| `--phase3_lambda` | `0.0` | Weight on Phase 3 shape loss |
| `--patience` | `20` | Early stopping patience |

### Outputs

| File | Description |
|---|---|
| `NeuralNet_predictions_phaseaware_v1.xlsx` | Actual vs predicted VB and ΔVB per row |
| `model_metrics_summary_phaseaware_v1.xlsx` | MAE, RMSE, R² + full hyperparameter log |
| `feature_importance_shap.xlsx` | SHAP-based feature importance ranking |

---

## Results

| Model | Test R² | Test MAE (µm) | Test RMSE (µm) | Monotonicity |
|---|---|---|---|---|
| XGBoost (with cutting length) | 0.922 | 12.89 | 17.49 | Post-processing only |
| XGBoost + ΔVB + clip | 0.940 | 15.30 | 17.28 | Post-processing only |
| MLP direct VB (no physics) | 0.802 | 20.88 | 27.55 | None |
| Monotone NN (softplus, no phases) | 0.864 | 18.51 | 22.87 | **Architectural** |
| **Phase-aware NN (this repo)** | **0.864** | **18.51** | **22.88** | **Architectural** |

The NN achieves lower R² than tree models but produces physically consistent wear trajectories by construction — monotonicity holds for unseen conditions without any post-processing.

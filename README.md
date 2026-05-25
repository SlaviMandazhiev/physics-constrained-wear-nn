# Physics-Constrained Neural Network for Tool Wear Prediction

A two-stage pipeline for predicting tool flank wear (VB) during CNC turning from force sensor signals, developed as part of the **SPP 2402** priority research programme at RWTH Aachen University.

**Stage 1** builds a feature matrix from raw force signals.
**Stage 2** trains a physics-constrained neural network that predicts wear with monotonicity guaranteed by construction.

---

## Background

During turning, three force components (Fx, Fy, Fz) are recorded continuously at ~6400 Hz for every turning pass (*Drehen*). Tool flank wear (VB, in µm) is measured manually at sparse, irregular intervals. The challenge is to predict the full wear trajectory from the dense force signal, given only a handful of ground-truth VB measurements.

The model addresses this by:
- Predicting **wear increments** (ΔVB) rather than absolute wear
- Enforcing **monotonicity architecturally** via a shifted softplus activation — wear can never decrease, without any post-processing
- Adding **phase-aware loss terms** that penalise predictions deviating from the expected curve shape in each of the three classical wear phases (break-in, steady-state, catastrophic)

---

## Pipeline overview

```
Raw .tdms files  →  downsampled CSVs  →  [Stage 1]  →  feature matrix (.xlsx)
                                                               ↓
                                                         [Stage 2]
                                                               ↓
                                                    predictions + SHAP importance
```

**Stage 1** (`build_nn_feature_matrix_signal_thirds.py`):
Segments each force signal into its active cutting region, splits it into three equal thirds, extracts 9 statistical features per force axis per third (27 features total), and pairs each third with an interpolated VB value.

**Stage 2** (`NN_PhysicsConstrained_PhaseAware_v1.py`):
Trains a physics-constrained MLP on the feature matrix. The model predicts wear rate → converts to ΔVB via cutting length increment → reconstructs the VB curve by cumulative sum.

---

## Installation

```bash
pip install -r requirements.txt
```

For GPU support, install the appropriate PyTorch build from [pytorch.org](https://pytorch.org) before running pip.

---

## Stage 1 — Build the feature matrix

### Input data structure

Stage 1 expects two inputs:

**1. Run folders** — a root directory containing one subdirectory per tool sample, named following the pattern:
```
spp2402_abr_wp_<id>-<Reihe>-<Schichtsystem>_<sample>_<run_id>/
```
Each run folder must contain Drehen subfolders (`01-Drehen1`, `02-Drehen2`, ...), each holding a downsampled force signal CSV with columns `Fx`, `Fy`, `Fz` (or equivalent — column names are detected automatically).

**2. Interpolated VB file** (`VB_interpolated_all_samples_combined.xlsx`) — an Excel file with columns:

| Column | Description |
|---|---|
| `Reihe` | Test series number (integer) |
| `Schichtsystem` | Coating system label |
| `Sample_ID` | Sample identifier matching the run folder name |
| `cutting_length_m` | Cumulative cutting length in metres |
| `VB_fit_um` | Interpolated flank wear in µm |
| `is_original` | Boolean — `True` for measured points, `False` for interpolated |

The script uses only the non-original (interpolated) rows, assigning 3 per Drehen positionally.

### Usage

```bash
python build_nn_feature_matrix_signal_thirds.py \
    --runs_root /path/to/run/folders \
    --combined_vb_xlsx /path/to/VB_interpolated_all_samples_combined.xlsx \
    --output FEATURE_MATRIX_SIGNAL_THIRDS.xlsx
```

### Signal processing arguments

| Argument | Default | Description |
|---|---|---|
| `--threshold` | `150.0` | Force threshold (N) — samples below this are treated as non-cutting |
| `--min_segment_length` | `5000` | Minimum samples for a valid cutting segment |
| `--min_break_length` | `100` | Gaps shorter than this (samples) are bridged |
| `--trim_samples` | `1600` | Samples removed from each segment edge (~1 s at 1600 Hz) |

### Output

`FEATURE_MATRIX_SIGNAL_THIRDS.xlsx` — one row per signal third per Drehen (~3 × number of Drehen). Columns:

- Metadata: `Sample_ID`, `Reihe`, `Schichtsystem`, `Run_ID`, `Tool_Sample_Code`, `Signal_File`, `Drehen_Index`, `Drehen_Position`, `Third_In_Drehen`
- Target: `Schnittweg_m`, `VB_um`, `Delta_VB_um`
- Features: `fx_mean`, `fx_max`, `fx_min`, `fx_std`, `fx_kurtosis`, `fx_skewness`, `fx_area_under_curve`, `fx_fft_energy`, `fx_segment_length` (and the same 9 for `fy_` and `fz_`)

Skipped Drehen and runs are logged to `<output>_warnings.log`.

---

## Stage 2 — Train the physics-constrained model

### Input

The feature matrix produced by Stage 1, with one additional column added manually before training:

**`Verschleiss_Phase`** (integer 1/2/3) — the wear phase label for each row:
- `1` — break-in phase (rapid early wear, decreasing rate)
- `2` — steady-state phase (approximately linear wear)
- `3` — catastrophic phase (accelerating wear toward tool failure)

Rows with missing or NaN phase labels are assigned phase 0 and excluded from phase-specific loss terms. The model still trains and predicts normally without this column, but the phase losses will be inactive.

### Usage

```bash
python NN_PhysicsConstrained_PhaseAware_v1.py \
    --data FEATURE_MATRIX_SIGNAL_THIRDS.xlsx \
    --results_dir results/my_run \
    --split_by sample_id \
    --delta_lambda 0.1 \
    --phase1_lambda 0.1 \
    --phase2_lambda 0.1 \
    --phase3_lambda 0.1
```

### How the model works

```
x_t  →  MLP  →  z_t  →  shifted_softplus  →  rate_t (≥ 0)
                                                    ↓
                                         ΔVB_t = rate_t × Δs_t
                                                    ↓
                               VB(t) = VB(0) + Σ ΔVB_i   [cumsum]
```

**Loss function:**
```
L = MSE(VB_pred, VB_true)
  + δ         · MSE(ΔVB_pred, ΔVB_true)    # wear increment supervision
  + λ₁ · L₁  (Phase 1: VB ≈ a·log(s+1)+b)  # break-in shape
  + λ₂ · L₂  (Phase 2: VB ≈ a·s+b)         # steady-state shape
  + λ₃ · L₃  (Phase 3: log(VB) ≈ a·s+b)    # catastrophic shape
```

Phase losses use differentiable least squares (`torch.linalg.solve`) inside the compute graph — gradients propagate through the fit parameters back to the model weights.

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--data` | `FINAL_FEATURE_MATRIX.xlsx` | Path to the feature matrix |
| `--results_dir` | *(required)* | Output directory (created automatically) |
| `--split_by` | `schichtsystem` | `sample_id`, `schichtsystem`, or `run_number` |
| `--epochs` | `120` | Maximum training epochs |
| `--batch_size` | `16` | Samples per batch (one sample = one full wear trajectory) |
| `--hidden` | `128` | Hidden units per layer (2 layers) |
| `--dropout` | `0.1` | Dropout rate |
| `--lr` | `5e-4` | Adam learning rate |
| `--delta_lambda` | `0.1` | Weight on the ΔVB increment loss |
| `--phase1_lambda` | `0.0` | Weight on Phase 1 logarithmic shape loss |
| `--phase2_lambda` | `0.0` | Weight on Phase 2 linear shape loss |
| `--phase3_lambda` | `0.0` | Weight on Phase 3 exponential shape loss |
| `--patience` | `20` | Early stopping patience |
| `--run_number_col` | `FORCE_Run_Number` | Column name for `--split_by run_number` |

### Outputs

All outputs are saved in a timestamped subdirectory inside `--results_dir`:

| File | Description |
|---|---|
| `NeuralNet_predictions_phaseaware_v1.xlsx` | Per-row predictions: actual vs predicted VB and ΔVB |
| `model_metrics_summary_phaseaware_v1.xlsx` | MAE, RMSE, R² on test set + full hyperparameter log |
| `feature_importance_shap.xlsx` | SHAP-based feature importance ranking |

---

## Results

Evaluated on Werkstoff 1, split by sample ID, 3 interpolated VB points per measurement interval:

| Model | Test R² | Test MAE (µm) | Test RMSE (µm) | Monotonicity |
|---|---|---|---|---|
| XGBoost (with cutting length) | 0.922 | 12.89 | 17.49 | Post-processing only |
| XGBoost + ΔVB + clip (40-pt interp) | 0.940 | 15.30 | 17.28 | Post-processing only |
| MLP direct VB (no physics) | 0.802 | 20.88 | 27.55 | None |
| Monotone NN (softplus, no phases) | 0.864 | 18.51 | 22.87 | **Architectural** |
| **Phase-aware NN (this repo)** | **0.864** | **18.51** | **22.88** | **Architectural** |

The NN achieves lower R² than tree models but guarantees physically consistent wear trajectories by construction — monotonicity holds for unseen coating systems without any post-processing or clipping.

---

## Acknowledgements

This work was carried out as part of **SPP 2402** (Deutsche Forschungsgemeinschaft priority programme) at RWTH Aachen University.

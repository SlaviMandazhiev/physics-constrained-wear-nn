# Physics-Constrained Neural Network for Tool Wear Prediction

A two-stage pipeline for predicting tool flank wear (VB) during CNC turning from force sensor signals. The model enforces monotonically increasing wear by construction using a physics-constrained architecture with phase-aware loss terms.

For a full explanation of the methodology, signal processing, model architecture, and results, see the project report (PDF).

---

## Installation

```bash
pip install -r requirements.txt
```

For GPU support, install the appropriate PyTorch build from [pytorch.org](https://pytorch.org) before running pip.

---

## Pipeline

```
Raw force signal CSVs
        │
        ▼
[build_feature_matrix.py]  (uses signal_processing.py)
  Segments active cutting signal, splits into thirds,
  extracts 27 statistical features per turning pass.
        │
        ▼
[NN_PhysicsConstrained_PhaseAware_v1.py]
  Trains a physics-constrained MLP that predicts wear
  increments via shifted softplus — monotonicity guaranteed
  by architecture, not post-processing.
        │
        ▼
  Predictions + SHAP feature importance
```

**Stage 1 — build the feature matrix:**
```bash
python build_feature_matrix.py \
    --runs_root /path/to/run/folders \
    --combined_vb_xlsx /path/to/vb_interpolated.xlsx \
    --output FEATURE_MATRIX.xlsx
```

**Stage 2 — train the model:**
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

"""
Neural network training script — PhaseAware v1
Based on Neural_Network_PhysicsTrain_LoopFormat_v2.py

New in this version:
- Phase-specific shape losses using differentiable least-squares (Option A)
  Phase 1 (break-in)    : VB follows a logarithmic  curve — fit log(s+1) via lstsq
  Phase 2 (steady-state): VB follows a linear        curve — fit s via lstsq
  Phase 3 (catastrophic): log(VB) follows a linear   curve — fit s via lstsq
                          (equivalent to VB following an exponential curve)
- Requires a "Verschleiss_Phase" column (integer 1/2/3) in the feature matrix
- CLI flags: --phase1_lambda, --phase2_lambda, --phase3_lambda (default 0.0)
- u_shape_lambda and smooth_lambda retained for backward compatibility
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# run dir helper
def make_run_dir(base_dir: str = "results") -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{stamp}"
    run_dir.mkdir()
    return run_dir

# ΔVB creation helper

def add_delta_vb(
    df_in: pd.DataFrame,
    run_col: str = "Sample_ID",
    x_col: str = "Schnittweg_m",
    vb_col: str = "VB_um",
    delta_col: str = "Delta_VB_um",
    clip_negative: bool = True,
) -> pd.DataFrame:
    df_out = df_in.copy()
    df_out["_row_order"] = np.arange(len(df_out))
    df_out = df_out.sort_values([run_col, x_col, "_row_order"]).reset_index(drop=True)

    delta_series = df_out.groupby(run_col)[vb_col].diff()
    df_out[delta_col] = delta_series.fillna(0.0)

    if clip_negative:
        df_out[delta_col] = df_out[delta_col].clip(lower=0.0)

    df_out = df_out.sort_values("_row_order").drop(columns=["_row_order"]).reset_index(drop=True)
    return df_out

# Masked losses

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err2 = (pred - target) ** 2
    err2 = err2 * mask
    return err2.sum() / mask.sum().clamp_min(1.0)


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    err = F.huber_loss(pred, target, reduction="none", delta=delta)
    err = err * mask
    return err.sum() / mask.sum().clamp_min(1.0)

# phase-specific shape loss

def _fit_line_loss(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Fit y = X @ coeffs via normal equations (differentiable).
    Returns mean squared residual.
    X: (N, 2),  y: (N,)
    Requires N >= 2.
    """
    XtX = X.T @ X + 1e-6 * torch.eye(2, device=X.device, dtype=X.dtype)
    Xty = X.T @ y
    coeffs = torch.linalg.solve(XtX, Xty)   # (2,)  — fully differentiable
    y_fit = X @ coeffs
    return ((y - y_fit) ** 2).mean()


def phase_shape_loss(
    VB_pred: torch.Tensor,          # (B, T)
    schnittweg: list,               # list of B numpy arrays (metres), each length T_i
    phases: torch.Tensor,           # (B, T) int, values 0/1/2/3  (0 = padding / unknown)
    mask: torch.Tensor,             # (B, T)
    phase1_lambda: float,
    phase2_lambda: float,
    phase3_lambda: float,
) -> torch.Tensor:
    """
    Per-sample, per-phase differentiable shape penalty.

    Phase 1 (log)    : fit  VB       ~ a*log(s+1) + b
    Phase 2 (linear) : fit  VB       ~ a*s + b
    Phase 3 (exp)    : fit  log(VB)  ~ a*s + b   => VB ~ exp(a*s+b)
    """
    if phase1_lambda == 0.0 and phase2_lambda == 0.0 and phase3_lambda == 0.0:
        return torch.tensor(0.0, device=VB_pred.device)

    lambdas = {1: phase1_lambda, 2: phase2_lambda, 3: phase3_lambda}
    modes   = {1: "log", 2: "linear", 3: "exp"}

    total = torch.tensor(0.0, device=VB_pred.device)
    n_terms = 0

    B = VB_pred.shape[0]
    for i in range(B):
        T_i = int(mask[i].sum().item())
        if T_i < 2:
            continue

        vb_i = VB_pred[i, :T_i]                         # (T_i,)  has grad
        s_i  = torch.tensor(
            schnittweg[i][:T_i], device=VB_pred.device, dtype=torch.float32
        )                                                # (T_i,)  no grad needed
        p_i  = phases[i, :T_i]                          # (T_i,)  int

        for phase_id, mode in modes.items():
            lam = lambdas[phase_id]
            if lam == 0.0:
                continue

            ph_mask = (p_i == phase_id)
            n_ph = int(ph_mask.sum().item())
            if n_ph < 2:
                continue  # need at least 2 points for a 2-parameter fit

            s_ph  = s_i[ph_mask]
            vb_ph = vb_i[ph_mask]

            if mode == "log":
                # VB ~ a*log(s+1) + b
                X = torch.stack([torch.log(s_ph + 1.0), torch.ones_like(s_ph)], dim=1)
                y = vb_ph

            elif mode == "linear":
                # VB ~ a*s + b
                X = torch.stack([s_ph, torch.ones_like(s_ph)], dim=1)
                y = vb_ph

            else:  # exp
                # log(VB) ~ a*s + b  =>  VB ~ exp(a*s + b)
                X = torch.stack([s_ph, torch.ones_like(s_ph)], dim=1)
                y = torch.log(vb_ph.clamp_min(1e-3))

            loss_ph = _fit_line_loss(X, y)
            total   = total + lam * loss_ph
            n_terms += 1

    if n_terms == 0:
        return torch.tensor(0.0, device=VB_pred.device)
    return total / n_terms

# physics-constrained model

class MonotoneWearModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: int = 128,
        dropout: float = 0.0,
        positivity: str = "shifted_softplus",
        positivity_shift: float = 1e-3,
        last_bias_init: float = -5.0,
    ):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Linear(n_features, hidden),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

        self.vb0_global = nn.Parameter(torch.tensor(0.0))
        self.positivity = positivity
        self.positivity_shift = positivity_shift

        final_layer = self.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.constant_(final_layer.bias, last_bias_init)

    def positive_map(self, z: torch.Tensor) -> torch.Tensor:
        if self.positivity == "softplus":
            return F.softplus(z)
        if self.positivity == "shifted_softplus":
            return torch.clamp(F.softplus(z) - self.positivity_shift, min=0.0)
        if self.positivity == "relu":
            return F.relu(z)
        raise ValueError(f"Unknown positivity mapping: {self.positivity}")

    def forward(
        self,
        X: torch.Tensor,
        delta_s: torch.Tensor,
        mask: torch.Tensor,
        vb0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z    = self.net(X).squeeze(-1)
        rate = self.positive_map(z)
        dVB  = rate * delta_s * mask

        if vb0 is None:
            vb0_use = self.vb0_global.expand(X.shape[0])
        else:
            vb0_use = vb0.view(-1)

        VB_pred = vb0_use[:, None] + torch.cumsum(dVB, dim=1)
        return VB_pred, dVB, rate

# dataset

@dataclass
class RunItem:
    sample_id: str
    schichtsystem: str
    rei: object
    schnittweg: np.ndarray
    X: np.ndarray
    VB: np.ndarray
    dVB_true: np.ndarray
    delta_s: np.ndarray
    vb0: float
    phases: np.ndarray           # integer array, values 1/2/3  (0 = unknown/missing)


class RunsDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        run_col: str = "Sample_ID",
        x_col: str = "Schnittweg_m",
        x_col_metres: str = "Schnittweg_m",
        vb_col: str = "VB_um",
        dvb_col: str = "Delta_VB_um",
        schicht_col: str = "Schichtsystem",
        rei_col: str = "Reihe",
        phase_col: str = "Verschleiss_Phase",
        scaler: Optional[StandardScaler] = None,
    ):
        self.items: List[RunItem] = []
        for sample_id, g in df.groupby(run_col):
            g2 = g.sort_values(x_col)

            X = g2[feature_cols].to_numpy(dtype=np.float32)
            if scaler is not None:
                X = scaler.transform(X).astype(np.float32)

            s    = g2[x_col].to_numpy(dtype=np.float32)
            s_m  = g2[x_col_metres].to_numpy(dtype=np.float32)
            VB   = g2[vb_col].to_numpy(dtype=np.float32)
            dVB  = g2[dvb_col].to_numpy(dtype=np.float32)

            delta_s = np.zeros_like(s, dtype=np.float32)
            delta_s[1:] = np.maximum(0.0, s[1:] - s[:-1])

            vb0 = float(VB[0])
            sch = str(g2[schicht_col].iloc[0]) if schicht_col in g2.columns else ""
            rei = g2[rei_col].iloc[0] if rei_col in g2.columns else None

            # Phase labels — default to 0 (unknown) if column absent or NaN
            if phase_col in g2.columns:
                ph = g2[phase_col].fillna(0).to_numpy(dtype=np.int32)
            else:
                ph = np.zeros(len(g2), dtype=np.int32)

            self.items.append(RunItem(
                sample_id=str(sample_id),
                schichtsystem=sch,
                rei=rei,
                schnittweg=s_m,
                X=X,
                VB=VB,
                dVB_true=dVB,
                delta_s=delta_s,
                vb0=vb0,
                phases=ph,
            ))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RunItem:
        return self.items[idx]


def collate_runs(batch: List[RunItem]) -> Dict:
    B     = len(batch)
    T_max = max(item.X.shape[0] for item in batch)
    Fdim  = batch[0].X.shape[1]

    X         = np.zeros((B, T_max, Fdim), dtype=np.float32)
    VB        = np.zeros((B, T_max),       dtype=np.float32)
    dVB_true  = np.zeros((B, T_max),       dtype=np.float32)
    delta_s   = np.zeros((B, T_max),       dtype=np.float32)
    mask      = np.zeros((B, T_max),       dtype=np.float32)
    vb0       = np.zeros((B,),             dtype=np.float32)
    phases    = np.zeros((B, T_max),       dtype=np.int32)
    schnittweg_arr = np.zeros((B, T_max),  dtype=np.float32)

    lengths    = []
    sample_ids = []
    schicht    = []
    rei        = []
    schnittweg = []

    for i, item in enumerate(batch):
        T = item.X.shape[0]
        X[i, :T, :]      = item.X
        VB[i, :T]        = item.VB
        dVB_true[i, :T]  = item.dVB_true
        delta_s[i, :T]   = item.delta_s
        mask[i, :T]      = 1.0
        vb0[i]           = item.vb0
        phases[i, :T]    = item.phases
        schnittweg_arr[i, :T] = item.schnittweg

        lengths.append(T)
        sample_ids.append(item.sample_id)
        schicht.append(item.schichtsystem)
        rei.append(item.rei)
        schnittweg.append(item.schnittweg)

    return {
        "X":           torch.from_numpy(X),
        "VB":          torch.from_numpy(VB),
        "dVB_true":    torch.from_numpy(dVB_true),
        "delta_s":     torch.from_numpy(delta_s),
        "mask":        torch.from_numpy(mask),
        "vb0":         torch.from_numpy(vb0),
        "phases":      torch.from_numpy(phases),
        "lengths":     lengths,
        "sample_ids":  sample_ids,
        "schichtsystem": schicht,
        "rei":         rei,
        "schnittweg":  schnittweg,   # list of numpy arrays (metres)
    }

# SHAP feature importance (optional)

def compute_shap_importance(
    model: nn.Module,
    train_ds: "RunsDataset",
    test_ds:  "RunsDataset",
    feature_cols: List[str],
    device: str,
    n_background: int = 100,
) -> pd.DataFrame:
    """
    Use SHAP DeepExplainer to compute per-feature importance.

    The model's core network (self.net + positive_map) maps each timestep's
    feature vector (F,): wear rate scalar independently of the sequence.
    We therefore explain it as a plain tabular function

    Returns a DataFrame sorted by mean SHAP descending.
    """
    import shap

    def stack_features(ds: "RunsDataset") -> np.ndarray:
        return np.vstack([item.X for item in ds.items])   # (N, F)

    X_train_2d = stack_features(train_ds)
    X_test_2d  = stack_features(test_ds)

    # random background subset from training data
    rng   = np.random.default_rng(42)
    n_bg  = min(n_background, len(X_train_2d))
    bg_idx = rng.choice(len(X_train_2d), size=n_bg, replace=False)
    X_bg   = torch.tensor(X_train_2d[bg_idx], dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test_2d,           dtype=torch.float32, device=device)


    class _RateNet(nn.Module):
        def __init__(self, m: MonotoneWearModel):
            super().__init__()
            self.m = m
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.m.net(x).squeeze(-1)
            return self.m.positive_map(z).unsqueeze(-1)  # (N, 1)

    wrapper = _RateNet(model).to(device)
    wrapper.eval()

    explainer   = shap.GradientExplainer(wrapper, X_bg)
    shap_values = explainer.shap_values(X_test)   # (N_test, F) or list thereof
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values.squeeze(-1)      # (N_test, F, 1) → (N_test, F)

    mean_abs = np.abs(shap_values).mean(axis=0)   # (F,)
    importance_df = (
        pd.DataFrame({"Feature": feature_cols, "Mean_Abs_SHAP": mean_abs})
        .sort_values("Mean_Abs_SHAP", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["Relative_Importance_pct"] = (
        importance_df["Mean_Abs_SHAP"] / importance_df["Mean_Abs_SHAP"].sum() * 100
    )
    importance_df.insert(0, "Rank", np.arange(1, len(importance_df) + 1))
    return importance_df

# prediction helper

@torch.no_grad()
def predict_on_loader(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []

    for batch in loader:
        X        = batch["X"].to(device)
        VB_true  = batch["VB"].to(device)
        dVB_true = batch["dVB_true"].to(device)
        delta_s  = batch["delta_s"].to(device)
        mask     = batch["mask"].to(device)
        vb0      = batch["vb0"].to(device)

        VB_pred, dVB_pred, rate_pred = model(X, delta_s, mask, vb0=vb0)

        VB_pred   = VB_pred.cpu().numpy()
        dVB_pred  = dVB_pred.cpu().numpy()
        rate_pred = rate_pred.cpu().numpy()
        VB_true   = VB_true.cpu().numpy()
        dVB_true  = dVB_true.cpu().numpy()
        phases_np = batch["phases"].numpy()

        for i in range(VB_pred.shape[0]):
            T   = batch["lengths"][i]
            sid = batch["sample_ids"][i]
            sch = batch["schichtsystem"][i]
            r   = batch["rei"][i]
            s   = batch["schnittweg"][i]

            for t in range(T):
                rows.append({
                    "Sample_ID":          sid,
                    "Schichtsystem":      sch,
                    "Reihe":              r,
                    "Schnittweg_m":       float(s[t]),
                    "Verschleiss_Phase":  int(phases_np[i, t]),
                    "Actual_VB":          float(VB_true[i, t]),
                    "Actual_Delta_VB":    float(dVB_true[i, t]),
                    "Predicted_Delta_VB": float(dVB_pred[i, t]),
                    "Predicted_VB":       float(VB_pred[i, t]),
                    "Predicted_Rate":     float(rate_pred[i, t]),
                })

    pred_df = pd.DataFrame(rows)
    pred_df = pred_df.sort_values(["Sample_ID", "Schnittweg_m"]).reset_index(drop=True)
    return pred_df

# train helper

def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    delta_lambda: float   = 0.1,
    smooth_lambda: float  = 0.0,
    u_shape_lambda: float = 0.0,
    phase1_lambda: float  = 0.0,
    phase2_lambda: float  = 0.0,
    phase3_lambda: float  = 0.0,
    patience: int         = 20,
    use_huber: bool       = False,
    huber_delta: float    = 1.0,
) -> Dict[str, float]:

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val   = float("inf")
    best_state = None
    best_epoch = epochs
    wait       = 0

    loss_fn = masked_huber if use_huber else masked_mse

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for batch in train_loader:
            X        = batch["X"].to(device)
            VB_true  = batch["VB"].to(device)
            dVB_true = batch["dVB_true"].to(device)
            delta_s  = batch["delta_s"].to(device)
            mask     = batch["mask"].to(device)
            vb0      = batch["vb0"].to(device)
            phases   = batch["phases"].to(device)

            optimizer.zero_grad()
            VB_pred, dVB_pred, rate_pred = model(X, delta_s, mask, vb0=vb0)

            if use_huber:
                loss_vb  = loss_fn(VB_pred,  VB_true,  mask, delta=huber_delta)
                loss_dvb = loss_fn(dVB_pred, dVB_true, mask, delta=huber_delta)
            else:
                loss_vb  = loss_fn(VB_pred,  VB_true,  mask)
                loss_dvb = loss_fn(dVB_pred, dVB_true, mask)

            loss = loss_vb + delta_lambda * loss_dvb

            # smoothness penalty
            if smooth_lambda > 0.0:
                d = dVB_pred[:, 1:] - dVB_pred[:, :-1]
                smooth = ((d ** 2) * mask[:, 1:]).sum() / mask[:, 1:].sum().clamp_min(1.0)
                loss = loss + smooth_lambda * smooth

            # u-shape penalty
            if u_shape_lambda > 0.0:
                d2     = rate_pred[:, 2:] - 2.0 * rate_pred[:, 1:-1] + rate_pred[:, :-2]
                mask2  = mask[:, 1:-1]
                u_pen  = (F.relu(-d2) ** 2 * mask2).sum() / mask2.sum().clamp_min(1.0)
                loss   = loss + u_shape_lambda * u_pen

            # phase-specific shape losses
            if phase1_lambda > 0.0 or phase2_lambda > 0.0 or phase3_lambda > 0.0:
                ph_loss = phase_shape_loss(
                    VB_pred       = VB_pred,
                    schnittweg    = batch["schnittweg"],
                    phases        = phases,
                    mask          = mask,
                    phase1_lambda = phase1_lambda,
                    phase2_lambda = phase2_lambda,
                    phase3_lambda = phase3_lambda,
                )
                loss = loss + ph_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running += float(loss.item())

        train_loss = running / max(1, len(train_loader))

        if val_loader is None:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f}")
            continue

        model.eval()
        v_running = 0.0
        with torch.no_grad():
            for batch in val_loader:
                X        = batch["X"].to(device)
                VB_true  = batch["VB"].to(device)
                dVB_true = batch["dVB_true"].to(device)
                delta_s  = batch["delta_s"].to(device)
                mask     = batch["mask"].to(device)
                vb0      = batch["vb0"].to(device)

                VB_pred, dVB_pred, _ = model(X, delta_s, mask, vb0=vb0)
                if use_huber:
                    loss_vb  = loss_fn(VB_pred,  VB_true,  mask, delta=huber_delta)
                    loss_dvb = loss_fn(dVB_pred, dVB_true, mask, delta=huber_delta)
                else:
                    loss_vb  = loss_fn(VB_pred,  VB_true,  mask)
                    loss_dvb = loss_fn(dVB_pred, dVB_true, mask)
                v_running += float((loss_vb + delta_lambda * loss_dvb).item())

        val_loss = v_running / max(1, len(val_loader))
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val - 1e-8:
            best_val   = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch:03d}. Best epoch: {best_epoch:03d}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_loss": best_val, "best_epoch": best_epoch}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data",         type=str,   default="FINAL_FEATURE_MATRIX.xlsx")
    p.add_argument("--results_dir",  type=str,   required=True,
                   help="Directory to save outputs into (created if it does not exist).")
    p.add_argument("--split_by",     type=str,   default="schichtsystem",
                   choices=["schichtsystem", "sample_id", "run_number"],
                   help="How to split train/test: by coating system (schichtsystem), individual sample (sample_id), or run number (run_number).")
    p.add_argument("--epochs",       type=int,   default=120)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--hidden",       type=int,   default=128)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--delta_lambda", type=float, default=0.1)
    p.add_argument("--smooth_lambda",  type=float, default=0.0)
    p.add_argument("--u_shape_lambda", type=float, default=0.0)
    
    # phase-specific shape loss weights
    p.add_argument("--phase1_lambda", type=float, default=0.0,
                   help="Weight for Phase 1 (break-in) logarithmic shape loss.")
    p.add_argument("--phase2_lambda", type=float, default=0.0,
                   help="Weight for Phase 2 (steady-state) linear shape loss.")
    p.add_argument("--phase3_lambda", type=float, default=0.0,
                   help="Weight for Phase 3 (catastrophic) exponential shape loss.")
    p.add_argument("--positivity",       type=str,   default="shifted_softplus",
                   choices=["softplus", "shifted_softplus", "relu"])
    p.add_argument("--positivity_shift", type=float, default=1e-3)
    p.add_argument("--last_bias_init",   type=float, default=-5.0)
    p.add_argument("--val_size",         type=float, default=0.2)
    p.add_argument("--patience",         type=int,   default=20)
    p.add_argument("--use_huber",        action="store_true")
    p.add_argument("--huber_delta",      type=float, default=1.0)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--normalize_schnittweg", action="store_true")
    p.add_argument("--run_number_col",   type=str,   default="FORCE_Run_Number",
                   help="Column name used when --split_by run_number is selected.")
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# main

def main():
    args = parse_args()
    set_seed(args.seed)

    df = pd.read_excel(args.data)

    vb_col     = "VB_um"
    dvb_col    = "Delta_VB_um"
    run_col    = "Sample_ID"
    x_col      = "Schnittweg_m"
    x_col_metres = "Schnittweg_m"
    schicht_col  = "Schichtsystem"
    phase_col    = "Verschleiss_Phase"

    df_clean = df.dropna(subset=[c for c in df.columns if c != phase_col],
                         how="any").reset_index(drop=True)
    df_clean = add_delta_vb(df_clean, run_col=run_col, x_col=x_col,
                             vb_col=vb_col, delta_col=dvb_col, clip_negative=True)

    if args.normalize_schnittweg:
        max_s = df_clean.groupby(run_col)[x_col].transform("max")
        df_clean["Schnittweg_m_normed"] = df_clean[x_col] / max_s
        x_col = "Schnittweg_m_normed"
        print("\nSchnittweg normalisation: ON")
    else:
        print("\nSchnittweg normalisation: OFF")

    # phase label summary
    if phase_col in df_clean.columns:
        counts = df_clean[phase_col].value_counts().sort_index()
        print(f"\nPhase label counts: {counts.to_dict()}")
    else:
        print(f"\nWARN: '{phase_col}' column not found — phase losses will be inactive.")

    # train / val / test split
    if args.split_by == "schichtsystem":
        groups = df_clean[schicht_col].unique()
        train_groups, test_groups = train_test_split(groups, test_size=0.2, random_state=args.seed)
        print(f"\nSplit by Schichtsystem")
        print(f"  Train: {sorted(train_groups)}")
        print(f"  Test : {sorted(test_groups)}")

        train_full_df = df_clean[df_clean[schicht_col].isin(train_groups)].reset_index(drop=True)
        test_df       = df_clean[df_clean[schicht_col].isin(test_groups)].reset_index(drop=True)
        train_only_groups, val_groups = train_test_split(train_groups, test_size=args.val_size, random_state=args.seed)
        train_df = train_full_df[train_full_df[schicht_col].isin(train_only_groups)].reset_index(drop=True)
        val_df   = train_full_df[train_full_df[schicht_col].isin(val_groups)].reset_index(drop=True)
        print(f"  Val  : {sorted(val_groups)}")

    elif args.split_by == "sample_id":
        sample_ids = df_clean[run_col].unique()
        train_ids, test_ids = train_test_split(sample_ids, test_size=0.2, random_state=args.seed)
        print(f"\nSplit by Sample_ID")
        print(f"  Total: {len(sample_ids)}  train: {len(train_ids)}  test: {len(test_ids)}")

        train_full_df = df_clean[df_clean[run_col].isin(train_ids)].reset_index(drop=True)
        test_df       = df_clean[df_clean[run_col].isin(test_ids)].reset_index(drop=True)
        train_only_ids, val_ids = train_test_split(train_ids, test_size=args.val_size, random_state=args.seed)
        train_df = train_full_df[train_full_df[run_col].isin(train_only_ids)].reset_index(drop=True)
        val_df   = train_full_df[train_full_df[run_col].isin(val_ids)].reset_index(drop=True)
        print(f"  Val  : {len(val_ids)} runs")

    else:  # run_number
        rn_col = args.run_number_col
        run_numbers = df_clean[rn_col].unique()
        train_rns, test_rns = train_test_split(run_numbers, test_size=0.2, random_state=args.seed)
        print(f"\nSplit by FORCE_Run_Number")
        print(f"  Total: {len(run_numbers)}  train: {len(train_rns)}  test: {len(test_rns)}")

        train_full_df = df_clean[df_clean[rn_col].isin(train_rns)].reset_index(drop=True)
        test_df       = df_clean[df_clean[rn_col].isin(test_rns)].reset_index(drop=True)
        train_only_rns, val_rns = train_test_split(train_rns, test_size=args.val_size, random_state=args.seed)
        train_df = train_full_df[train_full_df[rn_col].isin(train_only_rns)].reset_index(drop=True)
        val_df   = train_full_df[train_full_df[rn_col].isin(val_rns)].reset_index(drop=True)
        print(f"  Val  : {len(val_rns)} runs")

    # feature selection
    drop_cols = {
        vb_col, dvb_col,
        "Sample_ID", "Reihe", "Schichtsystem",
        "Run_ID", "Tool_Sample_Code", "Signal_File",
        "FORCE_Job_ID", "FORCE_Folder_Name", "FORCE_Run_Number",
        "AUCUSTIC_Job_ID", "AUCUSTIC_Folder_Name", "AUCUSTIC_Run_Number",
        "Drehen_Index", "Drehen_Position", "Point_In_Drehen", "Local_Progress_01",
        phase_col,          # phase labels are metadata, not features
        x_col,
        x_col_metres,
    }
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found after dropping metadata/targets.")

    print(f"\nFeature columns ({len(feature_cols)}):")
    for fc in feature_cols:
        print(f"  {fc}")

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].to_numpy(dtype=np.float32))

    ds_kwargs = dict(
        feature_cols=feature_cols,
        scaler=scaler,
        x_col=x_col,
        x_col_metres=x_col_metres,
        phase_col=phase_col,
    )
    train_ds = RunsDataset(train_df, **ds_kwargs)
    val_ds   = RunsDataset(val_df,   **ds_kwargs)
    test_ds  = RunsDataset(test_df,  **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_runs)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_runs)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_runs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MonotoneWearModel(
        n_features       = len(feature_cols),
        hidden           = args.hidden,
        dropout          = args.dropout,
        positivity       = args.positivity,
        positivity_shift = args.positivity_shift,
        last_bias_init   = args.last_bias_init,
    ).to(device)

    train_info = train_one_model(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        device         = device,
        epochs         = args.epochs,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        delta_lambda   = args.delta_lambda,
        smooth_lambda  = args.smooth_lambda,
        u_shape_lambda = args.u_shape_lambda,
        phase1_lambda  = args.phase1_lambda,
        phase2_lambda  = args.phase2_lambda,
        phase3_lambda  = args.phase3_lambda,
        patience       = args.patience,
        use_huber      = args.use_huber,
        huber_delta    = args.huber_delta,
    )

    pred_df = predict_on_loader(model, test_loader, device=device)

    delta_mae  = mean_absolute_error(pred_df["Actual_Delta_VB"], pred_df["Predicted_Delta_VB"])
    delta_rmse = np.sqrt(mean_squared_error(pred_df["Actual_Delta_VB"], pred_df["Predicted_Delta_VB"]))
    delta_r2   = r2_score(pred_df["Actual_Delta_VB"], pred_df["Predicted_Delta_VB"])
    vb_mae     = mean_absolute_error(pred_df["Actual_VB"], pred_df["Predicted_VB"])
    vb_rmse    = np.sqrt(mean_squared_error(pred_df["Actual_VB"], pred_df["Predicted_VB"]))
    vb_r2      = r2_score(pred_df["Actual_VB"], pred_df["Predicted_VB"])

    run_dir   = make_run_dir(base_dir=args.results_dir)
    pred_path = run_dir / "NeuralNet_predictions_phaseaware_v1.xlsx"
    pred_df.to_excel(pred_path, index=False)

    summary_df = pd.DataFrame([{
        "Model": "PyTorch_MonotoneWearModel_PhaseAware_v1",
        "Params": str({
            "split_by":        args.split_by,
            "epochs":          args.epochs,
            "batch_size":      args.batch_size,
            "hidden":          args.hidden,
            "dropout":         args.dropout,
            "lr":              args.lr,
            "weight_decay":    args.weight_decay,
            "delta_lambda":    args.delta_lambda,
            "smooth_lambda":   args.smooth_lambda,
            "u_shape_lambda":  args.u_shape_lambda,
            "phase1_lambda":   args.phase1_lambda,
            "phase2_lambda":   args.phase2_lambda,
            "phase3_lambda":   args.phase3_lambda,
            "positivity":      args.positivity,
            "positivity_shift":args.positivity_shift,
            "last_bias_init":  args.last_bias_init,
            "patience":        args.patience,
            "val_size":        args.val_size,
            "use_huber":       args.use_huber,
            "huber_delta":     args.huber_delta,
            "best_epoch":      train_info["best_epoch"],
            "best_val_loss":   train_info["best_val_loss"],
            "normalize_schnittweg": args.normalize_schnittweg,
        }),
        "Test_Delta_MAE":  delta_mae,
        "Test_Delta_RMSE": delta_rmse,
        "Test_Delta_R2":   delta_r2,
        "Test_VB_MAE":     vb_mae,
        "Test_VB_RMSE":    vb_rmse,
        "Test_VB_R2":      vb_r2,
        "Predictions_File": str(pred_path),
    }])

    summary_path = run_dir / "model_metrics_summary_phaseaware_v1.xlsx"
    summary_df.to_excel(summary_path, index=False)

    # SHAP feature importance
    print("\nComputing SHAP feature importance (this may take a moment)...")
    try:
        importance_df  = compute_shap_importance(
            model        = model,
            train_ds     = train_ds,
            test_ds      = test_ds,
            feature_cols = feature_cols,
            device       = device,
        )
        shap_path = run_dir / "feature_importance_shap.xlsx"
        importance_df.to_excel(shap_path, index=False)
        print("Feature importance saved:", shap_path)
        print("\nTop 10 most important features:")
        print(importance_df.head(10).to_string(index=False))
    except Exception as e:
        print(f"WARN: SHAP computation failed: {e}")
        shap_path = None

    print("\nDone.")
    print("Predictions:", pred_path)
    print("Summary:    ", summary_path)


if __name__ == "__main__":
    main()

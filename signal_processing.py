"""
signal_processing.py

General-purpose signal processing utilities for force sensor data
from CNC turning operations.

Provides:
  - Threshold-based active-cutting segmentation
  - Edge trimming and signal reconstruction
  - Statistical feature extraction per signal window
  - Flexible force column detection
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.stats import kurtosis, skew


# ── File I/O ────────────────────────────────────────────────────────────────

def read_table(path: Path) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


# ── Column detection ─────────────────────────────────────────────────────────

def _normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def detect_force_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Map canonical axis names ('fx', 'fy', 'fz') to actual DataFrame column names.
    Handles common variants (e.g. 'Messdaten_Fx', 'Force_X', 'FX').
    Returns a dict with only the axes that were found.
    """
    normalized = {_normalize_text(c): c for c in df.columns}
    out: Dict[str, str] = {}
    for canonical in ("fx", "fy", "fz"):
        if canonical in normalized:
            out[canonical] = normalized[canonical]
            continue
        for key, orig in normalized.items():
            if key.endswith(canonical) or key == f"force{canonical[-1]}":
                out[canonical] = orig
                break
    return out


# ── Signal segmentation ──────────────────────────────────────────────────────

def segment_active_signal(
    signal: np.ndarray,
    threshold: float = 150.0,
    min_segment_length: int = 5000,
    min_break_length: int = 100,
    trim_samples: int = 1600,
) -> np.ndarray:
    """
    Extract the steady-state cutting region from a raw force signal.

    Steps:
      1. Threshold — keep only samples where |signal| > threshold.
      2. Gap bridging — gaps shorter than min_break_length are merged.
      3. Length filter — segments shorter than min_segment_length are dropped.
      4. Edge trimming — trim_samples are removed from each segment boundary
         to exclude tool entry/exit transients.
      5. Concatenation — surviving segments are joined into one clean signal.

    Parameters
    ----------
    signal : 1-D array of raw force values (any unit consistent with threshold).
    threshold : Absolute force level that separates cutting from idle (default 150 N).
    min_segment_length : Minimum samples a segment must have after gap bridging.
    min_break_length : Gaps shorter than this (samples) are bridged, not split.
    trim_samples : Samples removed from each segment edge.

    Returns
    -------
    1-D float array — the concatenated clean signal, or empty array if nothing survives.
    """
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]
    if signal.size == 0:
        return np.array([], dtype=float)

    active_indices = np.flatnonzero(np.abs(signal) > threshold)
    if active_indices.size == 0:
        return np.array([], dtype=float)

    diffs = np.diff(active_indices)
    breaks = np.where(diffs > min_break_length)[0]
    starts = [active_indices[0]] + [active_indices[i + 1] for i in breaks]
    ends   = [active_indices[i] for i in breaks] + [active_indices[-1]]

    kept: List[np.ndarray] = []
    for start, end in zip(starts, ends):
        seg = signal[start:end + 1]
        if len(seg) <= min_segment_length:
            continue
        if len(seg) <= 2 * trim_samples:
            continue
        kept.append(seg[trim_samples:-trim_samples])

    if not kept:
        return np.array([], dtype=float)
    return np.concatenate(kept)


def split_into_thirds(signal: np.ndarray) -> List[np.ndarray]:
    """Split a 1-D signal into 3 non-overlapping, roughly equal parts."""
    n = len(signal)
    b = [0, n // 3, 2 * n // 3, n]
    return [signal[b[i]:b[i + 1]] for i in range(3)]


# ── Feature computation ──────────────────────────────────────────────────────

def _safe_skew(x: np.ndarray) -> float:
    if x.size < 3 or np.allclose(x, x[0]):
        return 0.0
    val = skew(x, bias=False)
    return 0.0 if not np.isfinite(val) else float(val)


def _safe_kurtosis(x: np.ndarray) -> float:
    if x.size < 4 or np.allclose(x, x[0]):
        return 0.0
    val = kurtosis(x, fisher=True, bias=False)
    return 0.0 if not np.isfinite(val) else float(val)


def compute_features(x: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Compute 9 statistical descriptors for a 1-D signal segment.

    Features (all prefixed with `prefix_`):
      segment_length, mean, max, min, std, kurtosis, skewness,
      area_under_curve (trapezoidal), fft_energy (sum of squared FFT magnitudes).

    Parameters
    ----------
    x : 1-D signal array (must be non-empty).
    prefix : Column name prefix, e.g. 'fx', 'fy', 'fz'.

    Returns
    -------
    Dict mapping '<prefix>_<feature>' to float value.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("Cannot compute features on an empty segment.")
    spec = np.abs(rfft(x))
    auc_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return {
        f"{prefix}_segment_length":   float(len(x)),
        f"{prefix}_mean":             float(np.mean(x)),
        f"{prefix}_max":              float(np.max(x)),
        f"{prefix}_min":              float(np.min(x)),
        f"{prefix}_std":              float(np.std(x)),
        f"{prefix}_kurtosis":         _safe_kurtosis(x),
        f"{prefix}_skewness":         _safe_skew(x),
        f"{prefix}_area_under_curve": float(auc_fn(x)),
        f"{prefix}_fft_energy":       float(np.sum(spec ** 2)),
    }

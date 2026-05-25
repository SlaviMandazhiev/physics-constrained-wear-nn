"""
build_nn_feature_matrix_signal_thirds.py

Builds an NN-ready feature matrix with 3 rows per Drehen by:
  1. Splitting each Drehen's active force signal into 3 non-overlapping thirds
  2. Computing statistical features on each third independently
  3. Matching each third to one of 3 interpolated (cutting_length, VB) points
     from VB_interpolated_all_samples_combined.xlsx

This gives 3 genuinely independent feature vectors per Drehen, each paired
with a different VB value along the monotone wear curve.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.stats import kurtosis, skew

# Naming patterns  (same as build_nn_feature_matrix_corrected_final.py)

OUTER_RUN_RE = re.compile(
    r"^spp2402_abr_wp_(?P<prefix>\d+)-(?P<reihe>Reihe-\d+)-(?P<system>(?:Mono|Bilay)-\d+)"
    r"_(?P<sample>[0-9]+-[0-9]+)_(?P<run_id>\d+)$",
    re.IGNORECASE,
)
INNER_PC_RE = re.compile(
    r"^spp2402_abr_wp_(?P<run_id>\d+)_pc_\d+_tf\d+$",
    re.IGNORECASE,
)
DREHEN_RE = re.compile(r"^(?P<prefix>\d+)-Drehen(?P<idx>\d+)$", re.IGNORECASE)


@dataclass
class RunMeta:
    run_id_str: str
    run_id_int: int
    reihe: str          # e.g. "Reihe-1"
    reihe_int: int      # e.g. 1
    schichtsystem: str  # e.g. "Mono-20"
    sample_code: str    # e.g. "121-1"
    run_dir: Path

    @property
    def sample_id(self) -> str:
        return f"{self.reihe}_{self.schichtsystem}_{self.sample_code}_run{self.run_id_str}"

    @property
    def vb_key(self) -> Tuple[int, str, str]:
        """Key to look up rows in the combined VB xlsx."""
        return (self.reihe_int, self.schichtsystem, self.sample_code)


# helpers

def normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


# folder walk

def find_run_dirs(runs_root: Path) -> List[RunMeta]:
    out: List[RunMeta] = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        m = OUTER_RUN_RE.match(child.name)
        if not m:
            continue
        reihe_str = m.group("reihe")          # "Reihe-1"
        reihe_num = int(reihe_str.split("-")[1])
        out.append(RunMeta(
            run_id_str=m.group("run_id"),
            run_id_int=int(m.group("run_id")),
            reihe=reihe_str,
            reihe_int=reihe_num,
            schichtsystem=m.group("system"),
            sample_code=m.group("sample"),
            run_dir=child,
        ))
    return out


def resolve_measurement_root(meta: RunMeta) -> Path:
    subdirs = [p for p in meta.run_dir.iterdir() if p.is_dir()]
    target_name = f"spp2402_abr_wp_{meta.run_id_str}_PC_001_TF1".lower()

    exact = [p for p in subdirs if p.name.lower() == target_name]
    if len(exact) == 1:
        return exact[0]

    pc_like = [p for p in subdirs
               if INNER_PC_RE.match(p.name) and
               int(INNER_PC_RE.match(p.name).group("run_id")) == meta.run_id_int]
    if len(pc_like) == 1:
        return pc_like[0]

    direct_drehen = [p for p in subdirs if DREHEN_RE.match(p.name)]
    if direct_drehen:
        return meta.run_dir

    if len(subdirs) == 1:
        return subdirs[0]

    raise FileNotFoundError(
        f"Could not resolve measurement root inside {meta.run_dir}."
    )


def find_drehen_dirs(meta: RunMeta) -> List[Tuple[int, Path]]:
    root = resolve_measurement_root(meta)
    found: List[Tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        m = DREHEN_RE.match(child.name)
        if m:
            found.append((int(m.group("idx")), child))
    return sorted(found, key=lambda t: t[0])


def find_signal_file(drehen_dir: Path) -> Path:
    candidates = [
        p for p in drehen_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".csv", ".xls", ".xlsx"}
    ]
    if not candidates:
        raise FileNotFoundError(f"No signal file in {drehen_dir}")
    ranked = []
    for p in candidates:
        name = p.name.lower()
        score = 5 if "downsampled" in name else (3 if "acoustic" in name else 0)
        score += 1 if p.suffix.lower() == ".csv" else 0
        ranked.append((score, p))
    ranked.sort(key=lambda t: (-t[0], t[1].name.lower()))
    return ranked[0][1]


# column detection

def detect_force_columns(df: pd.DataFrame) -> Dict[str, str]:
    normalized = {normalize_text(c): c for c in df.columns}
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

# Signal segmentation

def segment_active_signal(
    signal: np.ndarray,
    threshold: float = 150.0,
    min_segment_length: int = 5000,
    min_break_length: int = 100,
    trim_samples: int = 1600,
) -> np.ndarray:
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
    """Split signal into 3 non-overlapping, roughly equal parts."""
    n = len(signal)
    b = [0, n // 3, 2 * n // 3, n]
    return [signal[b[i]:b[i + 1]] for i in range(3)]

# Feature calculation

def safe_skew(x: np.ndarray) -> float:
    if x.size < 3 or np.allclose(x, x[0]):
        return 0.0
    val = skew(x, bias=False)
    return 0.0 if not np.isfinite(val) else float(val)


def safe_kurtosis(x: np.ndarray) -> float:
    if x.size < 4 or np.allclose(x, x[0]):
        return 0.0
    val = kurtosis(x, fisher=True, bias=False)
    return 0.0 if not np.isfinite(val) else float(val)


def compute_features(x: np.ndarray, prefix: str) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("Cannot compute features on an empty segment.")
    spec = np.abs(rfft(x))
    return {
        f"{prefix}_segment_length": float(len(x)),
        f"{prefix}_mean":           float(np.mean(x)),
        f"{prefix}_max":            float(np.max(x)),
        f"{prefix}_min":            float(np.min(x)),
        f"{prefix}_std":            float(np.std(x)),
        f"{prefix}_kurtosis":       safe_kurtosis(x),
        f"{prefix}_skewness":       safe_skew(x),
        f"{prefix}_area_under_curve": float(np.trapz(x) if hasattr(np, 'trapz') else np.trapezoid(x)),
        f"{prefix}_fft_energy":     float(np.sum(spec ** 2)),
    }

# VB lookup from combined xlsx

def load_combined_vb(path: Path) -> pd.DataFrame:
    """
    Load VB_interpolated_all_samples_combined.xlsx.
    Expected columns: Sample_ID, Reihe, Schichtsystem,
                      cutting_length_m, VB_fit_um, is_original, ...
    """
    df = pd.read_excel(path)
    return df


def get_all_drehen_vb_rows(
    sample_df: pd.DataFrame,
    n_drehen: int,
    rows_per_drehen: int = 3,
) -> List[pd.DataFrame]:
    """
    Assign non-original interpolated VB rows to each Drehen positionally.
    With endpoint=False in the interpolation, every interval contributes
    exactly rows_per_drehen non-original rows — so positional grouping
    is clean and robust regardless of stray extra rows in the xlsx.
    """
    non_orig = (
        sample_df[~sample_df["is_original"].astype(bool)]
        .sort_values("cutting_length_m")
        .reset_index(drop=True)
    )
    result = []
    for i in range(n_drehen):
        start = i * rows_per_drehen
        end   = (i + 1) * rows_per_drehen
        result.append(non_orig.iloc[start:end].reset_index(drop=True))
    return result

# processing

def build_rows_for_one_drehen(
    signal_file: Path,
    vb_rows: pd.DataFrame,      # 3 rows from combined xlsx
    meta: RunMeta,
    drehen_index: int,
    drehen_position: int,       # 0-indexed position within the sample
    threshold: float,
    min_segment_length: int,
    min_break_length: int,
    trim_samples: int,
) -> pd.DataFrame:

    raw = read_table(signal_file)
    force_cols = detect_force_columns(raw)
    if not {"fx", "fy", "fz"}.issubset(force_cols.keys()):
        raise ValueError(
            f"Missing force columns in {signal_file.name}. Detected: {force_cols}"
        )

    n_thirds = len(vb_rows)   # normally 3

    # for each axis: segment -> split into thirds -> features
    per_channel: Dict[str, List[Dict[str, float]]] = {}
    for axis in ("fx", "fy", "fz"):
        active = segment_active_signal(
            raw[force_cols[axis]].to_numpy(),
            threshold=threshold,
            min_segment_length=min_segment_length,
            min_break_length=min_break_length,
            trim_samples=trim_samples,
        )
        if active.size == 0:
            raise ValueError(f"Channel {axis} is empty after segmentation in {signal_file}")

        thirds = split_into_thirds(active)

        # If n_thirds != 3 for some edge case, adapt
        if n_thirds != 3:
            thirds = np.array_split(active, n_thirds)

        per_channel[axis] = [
            compute_features(thirds[i], prefix=axis)
            for i in range(n_thirds)
        ]

    out_rows: List[dict] = []
    for i in range(n_thirds):
        row = {
            "Sample_ID":         meta.sample_id,
            "Reihe":             meta.reihe,
            "Schichtsystem":     meta.schichtsystem,
            "Run_ID":            meta.run_id_int,
            "Tool_Sample_Code":  meta.sample_code,
            "Signal_File":       signal_file.name,
            "Drehen_Index":      drehen_index,
            "Drehen_Position":   drehen_position + 1,   # 1-indexed
            "Third_In_Drehen":   i + 1,                 # 1, 2, or 3
            "Schnittweg_m":      float(vb_rows.iloc[i]["cutting_length_m"]),
            "VB_um":             float(vb_rows.iloc[i]["VB_fit_um"]),
        }
        for axis in ("fx", "fy", "fz"):
            row.update(per_channel[axis][i])
        out_rows.append(row)

    return pd.DataFrame(out_rows)

# main pipeline

def build_feature_matrix(
    runs_root: Path,
    combined_vb_xlsx: Path,
    output_path: Path,
    threshold: float = 150.0,
    min_segment_length: int = 5000,
    min_break_length: int = 100,
    trim_samples: int = 1600,
) -> pd.DataFrame:

    run_dirs = find_run_dirs(runs_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run folders found in {runs_root}")

    print(f"Loading combined VB xlsx: {combined_vb_xlsx}")
    combined_vb = load_combined_vb(combined_vb_xlsx)

    vb_groups: Dict[Tuple[int, str, str], pd.DataFrame] = {}
    for (reihe, system, sid), grp in combined_vb.groupby(
        ["Reihe", "Schichtsystem", "Sample_ID"]
    ):
        vb_groups[(int(reihe), str(system), str(sid))] = grp.sort_values(
            "cutting_length_m"
        ).reset_index(drop=True)

    all_rows: List[pd.DataFrame] = []
    warnings: List[str] = []

    for meta in run_dirs:
        sample_df = vb_groups.get(meta.vb_key)
        if sample_df is None:
            warnings.append(
                f"Skipped {meta.run_dir.name}: no VB data for key {meta.vb_key}."
            )
            continue

        drehen_dirs = find_drehen_dirs(meta)
        if not drehen_dirs:
            warnings.append(f"Skipped {meta.run_dir.name}: no Drehen folders.")
            continue

        n_drehen = len(drehen_dirs)
        all_vb_rows = get_all_drehen_vb_rows(sample_df, n_drehen=n_drehen, rows_per_drehen=3)

        run_frames: List[pd.DataFrame] = []
        for pos in range(n_drehen):
            drehen_idx, drehen_dir = drehen_dirs[pos]
            try:
                vb_rows = all_vb_rows[pos]
                if len(vb_rows) == 0:
                    raise ValueError("No VB rows found for this Drehen (not enough interpolated points).")

                signal_file = find_signal_file(drehen_dir)
                print(f"      Processing {signal_file.name} ...")
                one = build_rows_for_one_drehen(
                    signal_file=signal_file,
                    vb_rows=vb_rows,
                    meta=meta,
                    drehen_index=drehen_idx,
                    drehen_position=pos,
                    threshold=threshold,
                    min_segment_length=min_segment_length,
                    min_break_length=min_break_length,
                    trim_samples=trim_samples,
                )
                run_frames.append(one)
                print(
                    f"OK  {meta.run_dir.name} | Drehen {drehen_idx:02d} "
                    f"| VB={vb_rows['VB_fit_um'].iloc[-1]:.1f}µm"
                )
            except Exception as exc:
                msg = f"Skipped {meta.run_dir.name} / {drehen_dir.name}: {exc}"
                warnings.append(msg)
                print(f"  WARN {msg}")

        if run_frames:
            run_df = pd.concat(run_frames, ignore_index=True)
            run_df = run_df.sort_values(
                ["Sample_ID", "Schnittweg_m", "Drehen_Index", "Third_In_Drehen"]
            ).reset_index(drop=True)
            run_df["Delta_VB_um"] = (
                run_df.groupby("Sample_ID")["VB_um"]
                .diff().fillna(0.0).clip(lower=0.0)
            )
            all_rows.append(run_df)

    if not all_rows:
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(" ", w)
        raise RuntimeError("No rows were built. Check the warnings above.")

    final_df = pd.concat(all_rows, ignore_index=True)
    final_df = final_df.sort_values(
        ["Sample_ID", "Schnittweg_m", "Drehen_Index", "Third_In_Drehen"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        final_df.to_csv(output_path, index=False)
    else:
        final_df.to_excel(output_path, index=False)

    if warnings:
        warn_path = output_path.with_name(output_path.stem + "_warnings.log")
        warn_path.write_text("\n".join(warnings), encoding="utf-8")
        print(f"\nWarnings written to: {warn_path}")

    print(f"\nSaved feature matrix: {output_path}")
    print(f"Shape: {final_df.shape}")
    return final_df

# entry point

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a 3-rows-per-Drehen NN feature matrix from raw force signal files."
    )
    p.add_argument(
        "--runs_root", type=str, required=True,
        help="Root directory containing the run folders (spp2402_abr_wp_* subdirectories).",
    )
    p.add_argument(
        "--combined_vb_xlsx", type=str, required=True,
        help="Path to VB_interpolated_all_samples_combined.xlsx.",
    )
    p.add_argument(
        "--output", type=str, default="FEATURE_MATRIX_SIGNAL_THIRDS.xlsx",
        help="Output path for the feature matrix (.xlsx or .csv). Default: FEATURE_MATRIX_SIGNAL_THIRDS.xlsx",
    )
    p.add_argument("--threshold",          type=float, default=150.0,
                   help="Force threshold (N) for active-cutting detection. Default: 150.0")
    p.add_argument("--min_segment_length", type=int,   default=5000,
                   help="Minimum samples for a valid cutting segment. Default: 5000")
    p.add_argument("--min_break_length",   type=int,   default=100,
                   help="Gaps shorter than this (samples) are bridged. Default: 100")
    p.add_argument("--trim_samples",       type=int,   default=1600,
                   help="Samples removed from each segment edge. Default: 1600")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_feature_matrix(
        runs_root=Path(args.runs_root),
        combined_vb_xlsx=Path(args.combined_vb_xlsx),
        output_path=Path(args.output),
        threshold=args.threshold,
        min_segment_length=args.min_segment_length,
        min_break_length=args.min_break_length,
        trim_samples=args.trim_samples,
    )

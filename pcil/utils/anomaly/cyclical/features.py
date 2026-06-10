"""
Cyclical anomaly pipeline — Step 2: per-cycle feature extraction
=================================================================
Three feature methods are implemented. The active method is controlled
by the FEATURE_METHOD constant or overridden per-call via the `method` argument.

Methods
-------
  stats    — 15 summary statistics per cycle
  waveform — cycle resampled to N_WAVEFORM fixed points (default 100)
  fft      — first N_FFT magnitude coefficients (default 20)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_WAVEFORM: int = 100
N_FFT: int = 20

FEATURE_METHOD: str = "waveform"


def _safe_vals(cycle_df: pd.DataFrame, signal_column: str) -> np.ndarray | None:
    if len(cycle_df) < 2 or signal_column not in cycle_df.columns:
        return None
    vals = cycle_df[signal_column].to_numpy(dtype=float)
    if np.isnan(vals).all():
        return None
    if np.isnan(vals).any():
        vals = np.where(np.isnan(vals), np.nanmean(vals), vals)
    return vals


# ─────────────────────────────────────────────────────────────
# Method 1: summary statistics
# ─────────────────────────────────────────────────────────────

def _extract_stats(vals: np.ndarray) -> dict[str, float]:
    duration = float(len(vals))
    peak     = float(np.max(vals))
    trough   = float(np.min(vals))
    mean     = float(np.mean(vals))
    std      = float(np.std(vals))
    area     = float(np.trapezoid(vals) if hasattr(np, "trapezoid") else np.trapz(vals))
    start_v  = float(vals[0])
    end_v    = float(vals[-1])
    delta    = end_v - start_v
    slope    = delta / max(duration - 1.0, 1.0)
    diffs    = np.diff(vals)
    mad      = float(np.mean(np.abs(diffs))) if len(diffs) else 0.0
    mxd      = float(np.max(np.abs(diffs))) if len(diffs) else 0.0
    return {
        "peak": peak, "trough": trough, "mean": mean, "std": std,
        "integrated_area": area, "cycle_duration": duration,
        "range": peak - trough, "start_value": start_v, "end_value": end_v,
        "delta": float(delta), "slope": float(slope),
        "area_per_sample": area / max(duration, 1.0),
        "mean_abs_diff": mad, "max_abs_diff": mxd,
        "time_to_peak_ratio": float(np.argmax(vals) / max(len(vals) - 1, 1)),
    }

def _zero_stats() -> dict[str, float]:
    return {k: 0.0 for k in ["peak","trough","mean","std","integrated_area",
        "cycle_duration","range","start_value","end_value","delta","slope",
        "area_per_sample","mean_abs_diff","max_abs_diff","time_to_peak_ratio"]}


# ─────────────────────────────────────────────────────────────
# Method 2: resampled waveform
# ─────────────────────────────────────────────────────────────

def _extract_waveform(vals: np.ndarray, n: int = N_WAVEFORM) -> dict[str, float]:
    resampled = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(vals)), vals)
    return {f"w{i:03d}": float(v) for i, v in enumerate(resampled)}

def _zero_waveform(n: int = N_WAVEFORM) -> dict[str, float]:
    return {f"w{i:03d}": 0.0 for i in range(n)}


# ─────────────────────────────────────────────────────────────
# Method 3: FFT magnitude coefficients
# ─────────────────────────────────────────────────────────────

def _extract_fft(vals: np.ndarray, n: int = N_FFT) -> dict[str, float]:
    coeffs = np.abs(np.fft.rfft(vals)) / max(len(vals), 1)
    if len(coeffs) < n:
        coeffs = np.pad(coeffs, (0, n - len(coeffs)))
    return {f"fft{i:03d}": float(coeffs[i]) for i in range(n)}

def _zero_fft(n: int = N_FFT) -> dict[str, float]:
    return {f"fft{i:03d}": 0.0 for i in range(n)}


# ─────────────────────────────────────────────────────────────
# public API
# ─────────────────────────────────────────────────────────────

def extract_features(
    cycle_df: pd.DataFrame,
    *,
    signal_column: str = "signal_value",
    method: str | None = None,
) -> dict[str, float]:
    """
    Return a feature dict for one cycle.

    Parameters
    ----------
    cycle_df      : one cycle's rows, already sliced upstream.
    signal_column : column containing the signal values.
    method        : override FEATURE_METHOD. One of "stats", "waveform", "fft".
    """
    chosen = method or FEATURE_METHOD
    vals   = _safe_vals(cycle_df, signal_column)

    if chosen == "stats":
        return _extract_stats(vals) if vals is not None else _zero_stats()
    if chosen == "waveform":
        return _extract_waveform(vals) if vals is not None else _zero_waveform()
    if chosen == "fft":
        return _extract_fft(vals) if vals is not None else _zero_fft()

    raise ValueError(f"Unknown feature method: {chosen!r}. Choose stats / waveform / fft.")


def stack_features(per_cycle_dicts: list[dict[str, float]]) -> pd.DataFrame:
    """Stack a list of per-cycle feature dicts into one DataFrame."""
    return pd.DataFrame(per_cycle_dicts)
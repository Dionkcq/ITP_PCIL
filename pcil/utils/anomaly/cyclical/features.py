"""
Cyclical anomaly pipeline — Step 2: per-cycle feature extraction
=================================================================
All three feature sets are implemented here so they can be compared.
The active method used by train.py is controlled by the FEATURE_METHOD
constant at the bottom of this file.

Methods
-------
  stats     — compact summary statistics per cycle
  waveform  — cycle resampled to N_WAVEFORM fixed points
  fft       — first N_FFT magnitude coefficients of the FFT

The stats feature set is intentionally interpretable and is recommended
for the first implementation. The waveform feature set is useful when cycle
shape deformation matters more than simple scalar statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── tuneable constants ──────────────────────────────────────
N_WAVEFORM: int = 100
N_FFT: int = 20

# Change this to "waveform" or "fft" to switch methods.
FEATURE_METHOD: str = "waveform"


# ─────────────────────────────────────────────────────────────
# shared helper
# ─────────────────────────────────────────────────────────────

def _safe_vals(cycle_df: pd.DataFrame, signal_column: str) -> np.ndarray | None:
    """Extract signal as a float array; return None if too short or all-NaN."""
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
    """
    Interpretable scalar features describing one cycle.

    The original 6 features are retained:
      peak, trough, mean, std, integrated_area, cycle_duration

    Additional simple shape features are added:
      range, start_value, end_value, delta, slope, area_per_sample,
      mean_abs_diff, max_abs_diff, time_to_peak_ratio
    """
    duration = float(len(vals))
    peak = float(np.max(vals))
    trough = float(np.min(vals))
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    area = float(np.trapezoid(vals) if hasattr(np, "trapezoid") else np.trapz(vals))
    signal_range = peak - trough
    start_value = float(vals[0])
    end_value = float(vals[-1])
    delta = end_value - start_value
    slope = delta / max(duration - 1.0, 1.0)
    area_per_sample = area / max(duration, 1.0)

    diffs = np.diff(vals)
    if len(diffs) == 0:
        mean_abs_diff = 0.0
        max_abs_diff = 0.0
    else:
        mean_abs_diff = float(np.mean(np.abs(diffs)))
        max_abs_diff = float(np.max(np.abs(diffs)))

    # Normalised peak position. 0 means peak near start, 1 means peak near end.
    time_to_peak_ratio = float(np.argmax(vals) / max(len(vals) - 1, 1))

    return {
        "peak": peak,
        "trough": trough,
        "mean": mean,
        "std": std,
        "integrated_area": area,
        "cycle_duration": duration,
        "range": float(signal_range),
        "start_value": start_value,
        "end_value": end_value,
        "delta": float(delta),
        "slope": float(slope),
        "area_per_sample": float(area_per_sample),
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
        "time_to_peak_ratio": time_to_peak_ratio,
    }


def _zero_stats() -> dict[str, float]:
    """Zero-filled fallback for invalid cycles."""
    return {
        "peak": 0.0,
        "trough": 0.0,
        "mean": 0.0,
        "std": 0.0,
        "integrated_area": 0.0,
        "cycle_duration": 0.0,
        "range": 0.0,
        "start_value": 0.0,
        "end_value": 0.0,
        "delta": 0.0,
        "slope": 0.0,
        "area_per_sample": 0.0,
        "mean_abs_diff": 0.0,
        "max_abs_diff": 0.0,
        "time_to_peak_ratio": 0.0,
    }


# ─────────────────────────────────────────────────────────────
# Method 2: resampled waveform
# ─────────────────────────────────────────────────────────────

def _extract_waveform(vals: np.ndarray, n: int = N_WAVEFORM) -> dict[str, float]:
    """
    Linearly interpolate the cycle to exactly `n` samples.
    Preserves waveform shape so the model can detect deformations.
    """
    x_old = np.linspace(0, 1, len(vals))
    x_new = np.linspace(0, 1, n)
    resampled = np.interp(x_new, x_old, vals)

    return {f"w{i:03d}": float(v) for i, v in enumerate(resampled)}


def _zero_waveform(n: int = N_WAVEFORM) -> dict[str, float]:
    return {f"w{i:03d}": 0.0 for i in range(n)}


# ─────────────────────────────────────────────────────────────
# Method 3: FFT magnitude coefficients
# ─────────────────────────────────────────────────────────────

def _extract_fft(vals: np.ndarray, n: int = N_FFT) -> dict[str, float]:
    """
    Take the real FFT of the cycle and keep the first `n` magnitude coefficients.

    Magnitudes are normalised by cycle length so short and long cycles produce
    comparable values.
    """
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
    Return a dict mapping feature_name -> float for one cycle.

    Parameters
    ----------
    cycle_df
        One cycle's rows, already sliced upstream.
    signal_column
        Column containing the signal values.
    method
        Override FEATURE_METHOD for this call.
        One of "stats", "waveform", "fft".
    """
    chosen = method or FEATURE_METHOD
    vals = _safe_vals(cycle_df, signal_column)

    if chosen == "stats":
        return _extract_stats(vals) if vals is not None else _zero_stats()
    if chosen == "waveform":
        return _extract_waveform(vals) if vals is not None else _zero_waveform()
    if chosen == "fft":
        return _extract_fft(vals) if vals is not None else _zero_fft()

    raise ValueError(
        f"Unknown feature method: {chosen!r}. "
        "Choose 'stats', 'waveform', or 'fft'."
    )


def stack_features(per_cycle_dicts: list[dict[str, float]]) -> pd.DataFrame:
    """Stack a list of per-cycle feature dicts into one DataFrame."""
    return pd.DataFrame(per_cycle_dicts)

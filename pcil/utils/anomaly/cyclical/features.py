"""
Cyclical anomaly pipeline — Step 2: per-cycle feature extraction
=================================================================
Converts each variable-length cycle into a fixed-size feature vector
by resampling the raw waveform to exactly N_WAVEFORM points.

Design choice: resampled waveform
-----------------------------------
Each cycle is linearly interpolated to 100 samples regardless of its
original length. This preserves the full pressure curve shape so the
1D CNN autoencoder can detect deformations (flattened peak, noise,
clipped top) that summary statistics would miss.

Winardi's recommendation: feed the entire curve directly to the model
rather than extracting secondary features, as this produces better
accuracy for the 1D CNN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_WAVEFORM: int = 100


def _safe_vals(cycle_df: pd.DataFrame, signal_column: str) -> np.ndarray | None:
    """Extract signal as float array; return None if too short or all-NaN."""
    if len(cycle_df) < 2 or signal_column not in cycle_df.columns:
        return None
    vals = cycle_df[signal_column].to_numpy(dtype=float)
    if np.isnan(vals).all():
        return None
    if np.isnan(vals).any():
        vals = np.where(np.isnan(vals), np.nanmean(vals), vals)
    return vals


def extract_features(
    cycle_df: pd.DataFrame,
    *,
    signal_column: str = "signal_value",
) -> dict[str, float]:
    """
    Resample one cycle to N_WAVEFORM points and return as a dict.

    Parameters
    ----------
    cycle_df      : one cycle's rows, already sliced upstream.
    signal_column : column containing the signal values.

    Returns
    -------
    dict with keys w000 … w099. Returns zeros if cycle is invalid.
    """
    vals = _safe_vals(cycle_df, signal_column)

    if vals is None:
        return {f"w{i:03d}": 0.0 for i in range(N_WAVEFORM)}

    x_old = np.linspace(0, 1, len(vals))
    x_new = np.linspace(0, 1, N_WAVEFORM)
    resampled = np.interp(x_new, x_old, vals)

    return {f"w{i:03d}": float(v) for i, v in enumerate(resampled)}


def stack_features(per_cycle_dicts: list[dict[str, float]]) -> pd.DataFrame:
    """Stack a list of per-cycle feature dicts into one DataFrame."""
    return pd.DataFrame(per_cycle_dicts)
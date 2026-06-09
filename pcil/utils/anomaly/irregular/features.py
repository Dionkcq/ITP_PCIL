"""
Irregular anomaly pipeline — Step 2: per-window feature extraction
===================================================================
Converts each variable-length window into a fixed-size feature vector.

Design choice: arrival-pattern features
----------------------------------------
With irregular sampling, the information lives in two places:

  1. WHEN events arrive — event rate and inter-arrival gaps. A burst
     (sensor flapping, repeated error flags) compresses the gaps; a
     stall (machine stopped reporting) stretches them.
  2. WHAT the events say — value statistics, when a value column
     exists at all (pure event logs may not have one).

Summary statistics are used instead of the cyclical pipeline's
resampled-waveform approach because irregular data has no curve shape
to preserve — interpolating a 3-event window to 100 points would
manufacture structure that was never measured.

Empty / single-event windows
-----------------------------
event_count carries the signal (0 or 1) and the gap features saturate
at window_seconds — "the silence lasted the whole window". Value
features fall back to 0.0; after per-machine z-score normalisation a
machine whose values never hover near 0 will see these windows stand
out, which is the desired behaviour for a reporting outage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_features(
    window_df: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    value_column: str | None = None,
    window_seconds: float,
) -> dict[str, float]:
    """
    Return a fixed-size feature dict for one window.

    Parameters
    ----------
    window_df        : the window's rows (may be EMPTY — see module doc).
    timestamp_column : column with event timestamps.
    value_column     : optional numeric column; adds value_* features.
    window_seconds   : window duration — saturation value for gap
                       features in sparse windows.

    Returns
    -------
    dict with keys:
      event_count, mean_interval, std_interval, max_interval
      (+ value_mean, value_std, value_min, value_max if value_column)
    """
    n = len(window_df)

    if n >= 2:
        ts = pd.to_datetime(window_df[timestamp_column])
        gaps = np.diff(ts.astype("int64").to_numpy()) / 1e9
        mean_interval = float(np.mean(gaps))
        std_interval = float(np.std(gaps))
        max_interval = float(np.max(gaps))
    else:
        # 0 or 1 events: no measurable gap — saturate at the window length.
        mean_interval = float(window_seconds)
        std_interval = 0.0
        max_interval = float(window_seconds)

    features: dict[str, float] = {
        "event_count": float(n),
        "mean_interval": mean_interval,
        "std_interval": std_interval,
        "max_interval": max_interval,
    }

    if value_column is not None:
        if n > 0:
            if value_column not in window_df.columns:
                raise ValueError(f"Missing value column: {value_column!r}")
            vals = window_df[value_column].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
        else:
            vals = np.array([])
        if len(vals) > 0:
            features["value_mean"] = float(np.mean(vals))
            features["value_std"] = float(np.std(vals))
            features["value_min"] = float(np.min(vals))
            features["value_max"] = float(np.max(vals))
        else:
            features["value_mean"] = 0.0
            features["value_std"] = 0.0
            features["value_min"] = 0.0
            features["value_max"] = 0.0

    return features


def stack_features(per_window_dicts: list[dict[str, float]]) -> pd.DataFrame:
    """Stack a list of per-window feature dicts into one DataFrame."""
    if not per_window_dicts:
        raise ValueError("No feature rows to stack — is the input DataFrame empty?")
    return pd.DataFrame(per_window_dicts)

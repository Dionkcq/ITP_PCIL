"""
Cyclical anomaly pipeline — Step 1: cycle detection
====================================================
Cuts the raw signal stream into individual cycles using peak detection.

Design choice: peak detection
------------------------------
SetPressure rises to a clear maximum and falls back down with each print
cycle. Peak detection finds these local maxima and treats the segment
between two consecutive peaks as one cycle.

Data note
---------
The logger runs at 1 kHz but the source samples at ~250 Hz, so values
repeat in 4-row groups. The signal is decimated to every 4th row before
detection to avoid detecting the same peak multiple times.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def _decimate(signal: np.ndarray, step: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Keep every `step`-th sample. Returns (decimated_signal, original_indices)."""
    if len(signal) == 0:
        return signal, np.array([], dtype=int)
    orig_idx = np.arange(0, len(signal), step)
    return signal[orig_idx], orig_idx


def detect_cycles(
    df: pd.DataFrame,
    *,
    signal_column: str = "signal_value",
    timestamp_column: str = "timestamp",
) -> Iterator[tuple[int, int]]:
    """
    Yield (start_idx, end_idx) tuples — one per detected cycle.

    Parameters
    ----------
    df            : DataFrame sorted by timestamp, single-machine slice.
    signal_column : column containing the cyclical signal.
    timestamp_column : kept for API compatibility, not used internally.
    """
    if signal_column not in df.columns:
        raise ValueError(f"Missing signal column: {signal_column!r}")

    signal = df[signal_column].to_numpy(dtype=float)

    if len(signal) < 8:
        return

    dec, orig_idx = _decimate(signal)

    if len(dec) < 3:
        return

    sig_range = float(np.nanmax(dec) - np.nanmin(dec))
    if not np.isfinite(sig_range) or sig_range < 1e-6:
        return

    peaks_dec, _ = find_peaks(
        dec,
        prominence=0.30 * sig_range,
        distance=50,               # minimum 50 decimated samples (~200 ms) between peaks
    )

    if len(peaks_dec) < 2:
        return

    peak_orig = orig_idx[peaks_dec]
    for i in range(len(peak_orig) - 1):
        start = int(peak_orig[i])
        end   = int(peak_orig[i + 1])
        if end - start >= 200:     # discard cycles shorter than 200 rows
            yield start, end
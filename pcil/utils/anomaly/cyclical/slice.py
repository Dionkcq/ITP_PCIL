"""
Cyclical anomaly pipeline — Step 1: cycle detection
====================================================
All three detection methods are implemented here so they can be compared.
The active method used by train.py is controlled by the SLICE_METHOD
constant at the bottom of this file.

Methods
-------
  peak           — find local maxima, yield peak-to-peak segments
  zero_crossing  — detrend the signal, find upward sign changes
  fixed_period   — chop the stream into equal-length windows

Data note
---------
The logger runs at 1 kHz but the source samples at about 250 Hz, so values
repeat in 4-row groups. All detection methods decimate to every 4th row
before boundary detection to avoid duplicate detections caused by repeated
values.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ─────────────────────────────────────────────────────────────
# shared helper
# ─────────────────────────────────────────────────────────────

def _decimate(signal: np.ndarray, step: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Keep every `step`-th sample. Returns (decimated_signal, original_indices)."""
    if len(signal) == 0:
        return signal, np.array([], dtype=int)

    orig_idx = np.arange(0, len(signal), step)
    return signal[orig_idx], orig_idx


def _remove_too_short_cycles(
    cycles: list[tuple[int, int]],
    min_len: int,
) -> list[tuple[int, int]]:
    """Remove cycle windows that are too short to be meaningful."""
    return [(int(s), int(e)) for s, e in cycles if int(e) - int(s) >= min_len]


# ─────────────────────────────────────────────────────────────
# Method 1: peak detection
# ─────────────────────────────────────────────────────────────

def _detect_peak(
    signal: np.ndarray,
    prominence_frac: float = 0.30,
    min_distance: int = 50,
    min_cycle_samples: int = 200,
) -> list[tuple[int, int]]:
    """
    Find local maxima on the decimated signal and map them back to original indices.

    prominence_frac
        Fraction of the decimated signal range used as prominence threshold.
    min_distance
        Minimum distance between peaks on the decimated signal.
        At 250 Hz, 50 samples = 0.2 seconds.
    min_cycle_samples
        Minimum cycle length on the original 1 kHz index scale.
    """
    dec, orig_idx = _decimate(signal)

    if len(dec) < 3:
        return []

    sig_range = float(np.nanmax(dec) - np.nanmin(dec))
    if not np.isfinite(sig_range) or sig_range < 1e-6:
        return []

    prominence = prominence_frac * sig_range

    peaks_dec, _ = find_peaks(
        dec,
        prominence=prominence,
        distance=min_distance,
    )

    if len(peaks_dec) < 2:
        return []

    peak_orig = orig_idx[peaks_dec]
    cycles = [
        (int(peak_orig[i]), int(peak_orig[i + 1]))
        for i in range(len(peak_orig) - 1)
    ]

    return _remove_too_short_cycles(cycles, min_cycle_samples)


# ─────────────────────────────────────────────────────────────
# Method 2: zero-crossing detection
# ─────────────────────────────────────────────────────────────

def _detect_zero_crossing(
    signal: np.ndarray,
    min_cycle_samples: int = 200,
) -> list[tuple[int, int]]:
    """
    Detrend the signal by subtracting a rolling mean, then find upward zero-crossings.

    This works best when the signal oscillates around a stable midpoint.
    It can over-detect when the baseline is noisy or the waveform is asymmetric.
    """
    dec, orig_idx = _decimate(signal)

    if len(dec) < 3:
        return []

    # Rolling window is a rough cycle-scale smoother.
    window = max(10, len(dec) // 20)
    rolled = (
        pd.Series(dec)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    detrended = dec - rolled

    signs = np.sign(detrended)
    crossings_dec = np.where((signs[:-1] < 0) & (signs[1:] >= 0))[0]

    if len(crossings_dec) < 2:
        return []

    crossing_orig = orig_idx[crossings_dec]
    cycles = [
        (int(crossing_orig[i]), int(crossing_orig[i + 1]))
        for i in range(len(crossing_orig) - 1)
    ]

    return _remove_too_short_cycles(cycles, min_cycle_samples)


# ─────────────────────────────────────────────────────────────
# Method 3: fixed-period window
# ─────────────────────────────────────────────────────────────

def _detect_fixed_period(
    signal: np.ndarray,
    period_ms: float = 2500.0,
    sample_rate_hz: float = 1000.0,
) -> list[tuple[int, int]]:
    """
    Chop the signal into equal-length windows.

    This is simple, but assumes the cycle period is known and stable.
    """
    rows_per_window = max(1, int(round(period_ms * sample_rate_hz / 1000.0)))
    n = len(signal)

    cycles = []
    for start in range(0, n - rows_per_window + 1, rows_per_window):
        end = start + rows_per_window
        cycles.append((int(start), int(end)))

    return cycles


# ─────────────────────────────────────────────────────────────
# public API
# ─────────────────────────────────────────────────────────────

# Change this to "zero_crossing" or "fixed_period" to switch methods.
SLICE_METHOD: str = "peak"


def detect_cycles(
    df: pd.DataFrame,
    *,
    signal_column: str = "signal_value",
    timestamp_column: str = "timestamp",
    method: str | None = None,
) -> Iterator[tuple[int, int]]:
    """
    Yield (start_idx, end_idx) tuples, one per detected cycle.

    Parameters
    ----------
    df
        DataFrame sorted by timestamp, usually for one machine.
    signal_column
        Column containing the cyclical signal.
    timestamp_column
        Kept for API compatibility. The current methods use row index.
    method
        Override SLICE_METHOD for this call.
        One of "peak", "zero_crossing", "fixed_period".
    """
    chosen = method or SLICE_METHOD

    if signal_column not in df.columns:
        raise ValueError(f"Missing signal column: {signal_column!r}")

    signal = df[signal_column].to_numpy(dtype=float)

    if len(signal) < 8:
        return

    if chosen == "peak":
        cycles = _detect_peak(signal)
    elif chosen == "zero_crossing":
        cycles = _detect_zero_crossing(signal)
    elif chosen == "fixed_period":
        cycles = _detect_fixed_period(signal)
    else:
        raise ValueError(
            f"Unknown slice method: {chosen!r}. "
            "Choose 'peak', 'zero_crossing', or 'fixed_period'."
        )

    yield from cycles

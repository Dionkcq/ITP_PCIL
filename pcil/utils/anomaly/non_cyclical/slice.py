"""
Why fixed windows instead of event-based slicing?
  Non-cyclical data (vibration, acoustic, temperature) has no natural
  repeat boundary. Fixed windows are the simplest, most predictable way
  to turn a stream into discrete observations a model can learn from.

Why 0.5 s as the default?
  At 25.6 kHz, 0.5 s = 12 800 rows — enough samples for RMS, kurtosis,
  and crest-factor to be statistically meaningful, while short enough
  that a transient fault won't be averaged away.

Generalisation note:
  This module is signal-agnostic. The same window logic works for any
  continuous sensor stream (temperature, current, pressure). Only the
  window_size_rows argument needs tuning for different sample rates or
  anomaly time-scales.
"""

from typing import Generator
import pandas as pd

def detect_windows(
    df: pd.DataFrame,
    *,
    window_size_rows: int,
    stride: int | None = None,
) -> Generator[tuple[int, int], None, None]:
    if window_size_rows <= 0:
        raise ValueError(f"window_size_rows must be positive, got {window_size_rows}")

    _stride = stride if stride is not None else window_size_rows

    if _stride <= 0:
        raise ValueError(f"stride must be positive, got {_stride}")

    n = len(df)
    start = 0

    while start + window_size_rows <= n:
        yield start, start + window_size_rows
        start += _stride

    # Trailing rows that don't fill a full window are silently dropped.
    # Rationale: a partial window produces unreliable statistics
    # (especially kurtosis), so it's safer to discard than to pad.
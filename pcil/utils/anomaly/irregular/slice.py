"""
Irregular anomaly pipeline — Step 1: time-based windowing
==========================================================
Cuts an irregularly-sampled stream into fixed-DURATION windows.

Why time-based instead of row-count windows?
---------------------------------------------
The cyclical pipeline slices on signal peaks; the non-cyclical pipeline
slices on a fixed number of rows. Both assume a uniform sample rate, so
"N rows" maps to a predictable span of wall-clock time. Irregular data
(event logs, error flags, sensors that report on-change) breaks that
assumption: 100 rows might cover 2 seconds or 2 hours. Fixed-duration
windows ("every 1.0 s of wall-clock") restore a consistent observation
unit regardless of how many events landed in it.

Why empty windows are KEPT (not dropped)
-----------------------------------------
For event-like data, silence is signal: a machine that stops emitting
heartbeats or sensor updates is often precisely the anomaly we want to
catch. An empty window therefore yields start_idx == end_idx rather
than being skipped, and features.py encodes it as event_count=0 with
gap features saturated at the window length.

Caveat: a long idle period (machine off overnight) produces one empty
window per stride. Slice the input to the period of interest before
scoring, or use a larger window for sparse data.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def timestamps_to_seconds(timestamps: pd.Series) -> np.ndarray:
    """Convert a timestamp column to float seconds since the first event.

    Accepts anything pd.to_datetime can parse (ISO strings, datetime64).
    The returned array is monotonically non-decreasing only if the input
    was sorted — callers sort by timestamp first.
    """
    ts = pd.to_datetime(timestamps)
    ns = ts.astype("int64").to_numpy()
    return (ns - ns[0]) / 1e9


def detect_windows(
    df: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    window_seconds: float = 1.0,
    stride_seconds: float | None = None,
) -> Iterator[tuple[pd.Timestamp, int, int]]:
    """
    Yield (window_start_time, start_idx, end_idx) per fixed-duration window.

    Parameters
    ----------
    df               : single-machine slice, sorted by timestamp,
                       0-based positional index (reset_index upstream).
    timestamp_column : column with event timestamps (parseable by
                       pd.to_datetime).
    window_seconds   : window duration in seconds of wall-clock time.
    stride_seconds   : step between window starts; defaults to
                       window_seconds (tumbling, non-overlapping windows).

    Yields
    ------
    (window_start_time, start_idx, end_idx) where df.iloc[start:end] is
    the window's rows. start_idx == end_idx for empty windows — the
    window start time is yielded explicitly because an empty window has
    no row to anchor a timestamp on.
    """
    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be positive, got {window_seconds}")
    stride = stride_seconds if stride_seconds is not None else window_seconds
    if stride <= 0:
        raise ValueError(f"stride_seconds must be positive, got {stride}")
    if timestamp_column not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column!r}")
    if len(df) == 0:
        return

    parsed = pd.to_datetime(df[timestamp_column])
    seconds = timestamps_to_seconds(df[timestamp_column])
    t0 = parsed.iloc[0]
    total = float(seconds[-1])

    start_t = 0.0
    while start_t <= total:
        end_t = start_t + window_seconds
        start_idx = int(np.searchsorted(seconds, start_t, side="left"))
        end_idx = int(np.searchsorted(seconds, end_t, side="left"))
        yield t0 + pd.to_timedelta(start_t, unit="s"), start_idx, end_idx
        start_t += stride

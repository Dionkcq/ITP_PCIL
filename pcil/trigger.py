"""
PCIL Trigger — generic time-slicer for tabular machine data
============================================================
Selects which rows of a DataFrame go downstream into preprocessing.
Two slice modes:
  - slice_by_time(df, start, end)   -> rows in [start, end]
  - slice_last_n_rows(df, n)        -> the most recent n rows

Imported by the orchestrator (`pcil/orchestrator.py`) via `_pull_slice()`.
The standalone CLI was removed in Week 3 — the orchestrator's
`/pipeline/run` and `/pipeline/save_csv` endpoints replace it. The
trigger parameters now live in `config.yaml` (the "recipe") instead of
being passed via CLI flags.

When the shop-floor database goes live, this same module will pull
slices from Postgres instead of accepting an in-memory DataFrame; the
function signatures should stay the same so callers don't break.
"""

from __future__ import annotations

from typing import Union

import pandas as pd

TimestampLike = Union[str, pd.Timestamp]


def slice_by_time(
    df: pd.DataFrame,
    start_time: TimestampLike,
    end_time: TimestampLike,
    *,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """
    Return rows where df[timestamp_column] is between start_time and end_time (inclusive on both ends).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a timestamp column.
    start_time, end_time : str | pd.Timestamp
        ISO 8601 strings or pandas Timestamps. UTC for now (pandas raises
        on mixed tz-aware / tz-naive comparisons — keep them consistent).
    timestamp_column : str, default "timestamp"

    Returns
    -------
    pd.DataFrame
        Filtered rows with a fresh 0-based index. Empty frame if start > end
        or no rows fall in the range.

    Raises
    ------
    KeyError
        If `timestamp_column` is not on the input DataFrame.
    """
    if timestamp_column not in df.columns:
        raise KeyError(
            f"slice_by_time: input DataFrame has no column '{timestamp_column}'. "
            f"Available columns: {list(df.columns)}"
        )

    # `format="ISO8601"` lets pandas accept mixed-precision ISO 8601 strings
    # in the same column (e.g. "...:18+00:00" alongside "...:17.094000+00:00").
    # Without it pandas locks onto the first row's exact format and rejects
    # anything that doesn't match.
    start = pd.to_datetime(start_time, format="ISO8601")
    end = pd.to_datetime(end_time, format="ISO8601")

    timestamps = pd.to_datetime(df[timestamp_column], format="ISO8601")
    mask = (timestamps >= start) & (timestamps <= end)
    return df.loc[mask].reset_index(drop=True)


def slice_last_n_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Return the most recent n rows of df.

    Parameters
    ----------
    df : pd.DataFrame
    n : int
        Number of rows. If n >= len(df), return the whole frame.
        If n <= 0, return an empty frame with the same columns.

    Returns
    -------
    pd.DataFrame
        With a fresh 0-based index.
    """
    if n <= 0:
        return df.iloc[0:0].reset_index(drop=True)
    return df.tail(n).reset_index(drop=True)

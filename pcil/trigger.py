"""
PCIL Trigger — generic time-slicer for tabular machine data
============================================================
Selects which rows of a DataFrame go downstream into preprocessing.
Two slice modes:
  - slice_by_time(df, start, end)   -> rows in [start, end]
  - slice_last_n_rows(df, n)        -> the most recent n rows

The CLI also writes the result to a CSV (default: triggered_slices/
slice_<run_time>.csv) so the next pipeline stage has a file to consume.

When the shop-floor database goes live, this same module will pull
slices from Postgres instead of accepting an in-memory DataFrame; the
function signatures should stay the same so callers don't break.

Run from PCIL_dev/:
    python pcil/trigger.py --input ../data/Clean_Data.csv --start "2025-08-08 10:08:18+00:00" --end "2025-08-08 10:08:19+00:00"
    python pcil/trigger.py --input ../data/Clean_Data.csv --last 1000
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import pandas as pd

TimestampLike = Union[str, pd.Timestamp]

DEFAULT_OUTPUT_DIR = "triggered_slices"


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


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _default_output_path() -> Path:
    """triggered_slices/slice_YYYYMMDDTHHMMSSZ.csv under the current working directory."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path.cwd() / DEFAULT_OUTPUT_DIR / f"slice_{stamp}.csv"


def main():
    parser = argparse.ArgumentParser(
        description="PCIL trigger — slice rows from a CSV and write the slice to a new CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python pcil/trigger.py --input ../data/Clean_Data.csv --start '2025-08-08 10:08:18+00:00' --end '2025-08-08 10:08:19+00:00'\n"
            "  python pcil/trigger.py --input ../data/Clean_Data.csv --last 1000\n"
            "  python pcil/trigger.py --input ../data/Clean_Data.csv --last 500 --output triggered_slices/my_slice.csv\n"
        ),
    )
    parser.add_argument("--input", required=True, help="CSV with a timestamp column.")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path. Defaults to "
            f"{DEFAULT_OUTPUT_DIR}/slice_<run_time>.csv under the current working directory. "
            "Parent directories are created if they don't exist."
        ),
    )
    parser.add_argument(
        "--sep",
        default=None,
        help=(
            "CSV delimiter. Defaults to auto-detect, so both comma-delimited "
            "shop-floor slices and semicolon-delimited Clean_Data.csv work."
        ),
    )
    parser.add_argument("--start", default=None, help="ISO 8601 start time (use with --end).")
    parser.add_argument("--end", default=None, help="ISO 8601 end time (use with --start).")
    parser.add_argument("--last", type=int, default=None, help="Return the last N rows.")
    parser.add_argument("--timestamp-column", default="timestamp",
                        help='Name of the timestamp column (default "timestamp").')
    parser.add_argument("--preview-rows", type=int, default=10,
                        help="How many rows to preview on stdout (default 10, set 0 to skip).")
    args = parser.parse_args()

    # Validate slice mode early
    if args.last is None and not (args.start and args.end):
        raise SystemExit("Pass either --start AND --end, or --last.")
    if args.last is not None and (args.start or args.end):
        raise SystemExit("Pass either --start/--end or --last, not both.")

    # Load input
    if args.sep is None:
        df = pd.read_csv(args.input, sep=None, engine="python")
    else:
        df = pd.read_csv(args.input, sep=args.sep)
    print(f"Loaded {len(df):,} rows from {args.input}")

    # Slice
    if args.last is not None:
        out = slice_last_n_rows(df, args.last)
        mode = f"last {args.last} rows"
    else:
        out = slice_by_time(df, args.start, args.end, timestamp_column=args.timestamp_column)
        mode = f"time range [{args.start}, {args.end}]"

    # Save
    out_path = Path(args.output) if args.output else _default_output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # Report
    print(f"Slice mode: {mode}")
    print(f"Returned   : {len(out):,} rows")
    print(f"Saved      -> {out_path}")
    if args.preview_rows > 0 and len(out) > 0:
        print()
        print("Preview:")
        print(out.head(args.preview_rows).to_string())


if __name__ == "__main__":
    main()

"""
Cyclical anomaly pipeline — Step 0: data prep
==============================================
Turn `data/Clean_Data.csv` into `cyclical_dataset.csv` (training) and
`cyclical_eval.csv` (held-out with synthetic anomalies + cycle_label).

Steps (per the menu task 1):
  1. Reformat: semicolon -> comma; rename _time -> timestamp,
     SetPressure -> signal_value. Drop SetVelo.
  2. Optional: fake a second machine via timestamp shift +
     small constant offset to signal_value.
  3. Hold out the last 20% per machine -> eval set; inject 5–10%
     synthetic anomalies into fixed 1-second windows; label them.
  4. Remaining 80% -> training set (no labels).

Run from PCIL_dev/:
    python -m pcil.utils.anomaly.cyclical.prepare_data \\
        --input ../data/Clean_Data.csv \\
        --output-dir ../data/

Produces:
    <output-dir>/cyclical_dataset.csv
    <output-dir>/cyclical_eval.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Step 1 — reformat
# ─────────────────────────────────────────────────────────────

def reformat(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up the raw `Clean_Data.csv`:
      - Rename `_time` -> `timestamp`, `SetPressure` -> `signal_value`.
      - Drop `SetVelo` (mostly constant in the recording).
      - Make sure `timestamp` is parsed as datetime.
    """
    df = raw.copy()

    # Rename columns
    df = df.rename(columns={
        "_time": "timestamp",
        "SetPressure": "signal_value",
    })

    # Drop SetVelo — constant at -10 throughout the entire recording
    df = df.drop(columns=["SetVelo"], errors="ignore")

    # Parse timestamp as datetime (file uses ISO 8601 with UTC offset)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────
# Step 2 — optional second machine
# ─────────────────────────────────────────────────────────────

def fake_second_machine(
    df: pd.DataFrame,
    *,
    offset: float = 0.05,
    time_shift: pd.Timedelta = pd.Timedelta(minutes=5),
) -> pd.DataFrame:
    """
    Duplicate rows with shifted time + nudged amplitude to test that the
    pipeline can carry separate machine baselines. Tag original as
    `inkjet_01`, duplicate as `inkjet_02`. Adds a `machine_id` column.

    This is only a local demo/stress test. In production, train a fitted
    anomaly bundle from real baseline data for each actual machine.
    """
    df = df.copy()
    df["machine_id"] = "inkjet_01"

    df2 = df.copy()
    df2["machine_id"] = "inkjet_02"
    df2["timestamp"] = df2["timestamp"] + time_shift
    df2["signal_value"] = df2["signal_value"] + offset

    combined = pd.concat([df, df2], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


# ─────────────────────────────────────────────────────────────
# Step 3 — inject synthetic anomalies for eval
# ─────────────────────────────────────────────────────────────

def inject_anomalies(
    eval_df: pd.DataFrame,
    *,
    fraction: float = 0.08,
    window_size_rows: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Split `eval_df` into fixed `window_size_rows` windows, distort
    `fraction` of them at random, and add a `cycle_label` column
    (`normal` / `anomalous`) repeated across rows in each window.

    Distortion types (varied so the detector sees multiple anomaly modes):
      - double_amplitude : amplitude *= 2 around the signal mean
      - clip_peak        : clip values above the 90th percentile
      - replace_noise    : replace with Gaussian noise (same mean/std)
      - stretch          : repeat every other sample (simulates slow cycle)
    """
    rng = np.random.default_rng(seed)
    df = eval_df.copy().reset_index(drop=True)

    # Assign each row to a window index
    window_ids = np.arange(len(df)) // window_size_rows
    n_windows = int(window_ids.max()) + 1

    # Pick which windows to distort
    n_anomalous = max(1, int(round(n_windows * fraction)))
    anomalous_windows = set(
        rng.choice(n_windows, size=n_anomalous, replace=False).tolist()
    )

    distortion_types = ["double_amplitude", "clip_peak", "replace_noise", "stretch"]
    labels = ["normal"] * len(df)

    for wid in range(n_windows):
        idx = np.where(window_ids == wid)[0]
        if len(idx) == 0 or wid not in anomalous_windows:
            continue

        kind = distortion_types[int(rng.integers(0, len(distortion_types)))]
        vals = df.loc[idx, "signal_value"].to_numpy(dtype=float)

        if kind == "double_amplitude":
            mid = vals.mean()
            vals = mid + 2.0 * (vals - mid)

        elif kind == "clip_peak":
            threshold = np.percentile(vals, 90)
            vals = np.clip(vals, None, threshold)

        elif kind == "replace_noise":
            vals = rng.normal(
                loc=vals.mean(), scale=vals.std() + 1e-6, size=len(vals)
            )

        elif kind == "stretch":
            # Repeat every other sample — simulates a slowed-down cycle
            stretched = np.repeat(vals[::2], 2)[: len(vals)]
            vals = stretched

        df.loc[idx, "signal_value"] = vals
        for i in idx:
            labels[i] = "anomalous"

    df["cycle_label"] = labels
    return df


# ─────────────────────────────────────────────────────────────
# CLI  (wiring already done in skeleton — kept identical)
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prep cyclical training + eval datasets from Clean_Data.csv.",
    )
    parser.add_argument("--input", required=True, help="Path to Clean_Data.csv.")
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write cyclical_dataset.csv + cyclical_eval.csv to.",
    )
    parser.add_argument(
        "--no-fake-machine", action="store_true",
        help="Skip step 2 (don't fake a second machine).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for anomaly injection."
    )
    args = parser.parse_args()

    # Auto-detect delimiter — spec says semicolon but actual file is comma-delimited
    raw = pd.read_csv(args.input, sep=None, engine="python")
    print(f"Loaded {len(raw):,} rows from {args.input}")

    cleaned = reformat(raw)
    print(f"After reformat: {len(cleaned):,} rows, columns: {list(cleaned.columns)}")

    if args.no_fake_machine:
        cleaned["machine_id"] = "inkjet_01"
    else:
        cleaned = fake_second_machine(cleaned)
        print("Added fake second machine (inkjet_02).")

    # Hold out the last 20% per machine for eval
    train_frames, eval_frames = [], []
    for mid, group in cleaned.groupby("machine_id"):
        n = len(group)
        cutoff = int(n * 0.8)
        train_frames.append(group.iloc[:cutoff])
        eval_frames.append(group.iloc[cutoff:])
    train_df = pd.concat(train_frames, ignore_index=True)
    eval_df  = pd.concat(eval_frames,  ignore_index=True)

    eval_df = inject_anomalies(eval_df, seed=args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "cyclical_dataset.csv"
    eval_path  = out_dir / "cyclical_eval.csv"
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path,   index=False)

    n_anomalous = int((eval_df["cycle_label"] == "anomalous").sum())
    print(f"Saved -> {train_path} ({len(train_df):,} rows)")
    print(f"Saved -> {eval_path}  ({len(eval_df):,} rows, {n_anomalous} anomalous rows)")


if __name__ == "__main__":
    main()
"""
Irregular anomaly pipeline — score CLI
=======================================
Takes a CSV slice + that machine's trained .pkl from train.py, reapplies
the same windowing + feature extraction + fitted per-machine normaliser,
and returns a per-window DataFrame with `anomaly_score` and `is_anomaly`
columns.

The threshold used for `is_anomaly` is stored in the .pkl bundle at
train time (95th percentile of training scores).

Run from repo root:
    python -m pcil.utils.anomaly.irregular.score \\
        --input data/irregular_eval.csv \\
        --model data/irregular_inkjet_01.pkl \\
        --output data/irregular_eval_scored.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pcil.utils.anomaly.irregular.slice import detect_windows
from pcil.utils.anomaly.irregular.features import extract_features, stack_features


def score(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """
    Return a DataFrame with one row per time window, containing the
    extracted features plus `window_start_timestamp`, `anomaly_score`
    and `is_anomaly` columns.

    When the input produces no windows, returns an empty DataFrame with
    the expected columns so callers see "0 windows scored", not an error.
    """
    machine_id_column = bundle["machine_id_column"]
    timestamp_column  = bundle["timestamp_column"]
    value_column      = bundle["value_column"]
    window_seconds    = bundle["window_seconds"]
    feature_columns   = bundle["feature_columns"]
    normaliser        = bundle["normaliser"]
    model             = bundle["model"]
    threshold         = bundle.get("threshold", None)

    rows = []
    for machine_id, group in df.groupby(machine_id_column):
        group = group.sort_values(timestamp_column).reset_index(drop=True)
        for w_start, start, end in detect_windows(
            group,
            timestamp_column=timestamp_column,
            window_seconds=window_seconds,
        ):
            features = extract_features(
                group.iloc[start:end],
                timestamp_column=timestamp_column,
                value_column=value_column,
                window_seconds=window_seconds,
            )
            features[machine_id_column] = machine_id
            features["window_start_timestamp"] = w_start
            rows.append(features)

    feature_df = stack_features(rows)

    if feature_df.empty:
        # Defensive: detect_windows keeps empty windows, so any non-empty
        # input yields at least one window — but keep the same graceful
        # empty-result contract as the cyclical/non_cyclical scorers.
        return pd.DataFrame(
            columns=[*feature_columns, machine_id_column,
                     "window_start_timestamp", "anomaly_score", "is_anomaly"]
        )

    feature_df_norm = normaliser.transform(feature_df, machine_id_column=machine_id_column)
    X               = feature_df_norm[feature_columns].to_numpy(dtype=float)
    scores          = model.score(X)

    feature_df["anomaly_score"] = scores

    # Apply threshold — fall back to median if not stored in bundle
    if threshold is None:
        threshold = float(np.median(scores))
    feature_df["is_anomaly"] = scores > threshold

    return feature_df


def main():
    parser = argparse.ArgumentParser(description="Score irregular data with a trained model.")
    parser.add_argument("--input",  required=True, help="CSV slice to score.")
    parser.add_argument("--model",  required=True, help="Path to .pkl from train.py.")
    parser.add_argument("--output", required=True, help="Where to save scored CSV.")
    args = parser.parse_args()

    df     = pd.read_csv(args.input)
    bundle = joblib.load(args.model)
    scored = score(df, bundle)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    print(f"Scored {len(scored)} windows.")
    print(f"Threshold used: {bundle.get('threshold', float('nan')):.4f}")
    print(f"Anomalies flagged: {scored['is_anomaly'].sum()} / {len(scored)}")


if __name__ == "__main__":
    main()

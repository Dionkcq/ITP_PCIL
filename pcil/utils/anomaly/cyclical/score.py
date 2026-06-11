"""
Cyclical anomaly pipeline — score CLI
======================================
Takes a CSV slice + that machine's trained .pkl from train.py, reapplies
the same slicing + feature extraction + fitted per-machine normaliser,
and returns a per-cycle DataFrame with `anomaly_score` and `is_anomaly` columns.

The threshold used for `is_anomaly` is stored in the .pkl bundle at train time
(95th percentile of training scores). When labelled eval data is available,
call find_best_threshold() to refine it and update the bundle.

Run from repo root:
    python -m pcil.utils.anomaly.cyclical.score \\
        --input data/cyclical_eval.csv \\
        --model data/inkjet_cyclical.pkl \\
        --output data/cyclical_eval_scored.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pcil.utils.anomaly.cyclical.slice import detect_cycles
from pcil.utils.anomaly.cyclical.features import extract_features, stack_features


def find_best_threshold(
    scores: np.ndarray,
    true_labels: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Sweep percentile thresholds and return the one that maximises F1.

    Parameters
    ----------
    scores      : anomaly scores from model.score()
    true_labels : array of "normal" / "anomalous" strings

    Returns
    -------
    (best_threshold, best_precision, best_recall, best_f1)
    """
    best_f1, best_thresh, best_p, best_r = 0.0, 0.0, 0.0, 0.0

    for pct in np.arange(1, 100, 1):
        thresh = float(np.percentile(scores, pct))
        pred   = np.where(scores > thresh, "anomalous", "normal")
        tp = ((pred == "anomalous") & (true_labels == "anomalous")).sum()
        fp = ((pred == "anomalous") & (true_labels == "normal")).sum()
        fn = ((pred == "normal")    & (true_labels == "anomalous")).sum()
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thresh, best_p, best_r = f1, thresh, p, r

    return best_thresh, best_p, best_r, best_f1


def score(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """
    Return a DataFrame with one row per detected cycle, containing the
    extracted features plus `anomaly_score` and `is_anomaly` columns.

    `is_anomaly` is True when anomaly_score exceeds the threshold stored
    in the bundle.
    """
    machine_id_column = bundle["machine_id_column"]
    signal_column     = bundle["signal_column"]
    timestamp_column  = bundle["timestamp_column"]
    feature_columns   = bundle["feature_columns"]
    normaliser        = bundle["normaliser"]
    model             = bundle["model"]
    threshold         = bundle.get("threshold", None)

    rows = []
    for machine_id, group in df.groupby(machine_id_column):
        group = group.sort_values(timestamp_column).reset_index(drop=True)
        for start, end in detect_cycles(
            group, signal_column=signal_column, timestamp_column=timestamp_column,
        ):
            features = extract_features(group.iloc[start:end], signal_column=signal_column)
            features[machine_id_column] = machine_id
            features["cycle_start_timestamp"] = group.iloc[start][timestamp_column]
            rows.append(features)

    feature_df = stack_features(rows)

    if feature_df.empty:
        # No complete cycle detected — input shorter than one ~200-row
        # cycle, or no clear peaks. Return an empty result with the
        # expected columns so callers see "0 cycles scored", not an error.
        return pd.DataFrame(
            columns=[*feature_columns, machine_id_column,
                     "cycle_start_timestamp", "anomaly_score", "is_anomaly"]
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
    parser = argparse.ArgumentParser(description="Score cyclical data with a trained model.")
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
    print(f"Scored {len(scored)} cycles.")
    print(f"Threshold used: {bundle.get('threshold', 'median fallback'):.4f}")
    print(f"Anomalies flagged: {scored['is_anomaly'].sum()} / {len(scored)}")


if __name__ == "__main__":
    main()
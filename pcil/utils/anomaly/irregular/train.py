"""
Irregular anomaly pipeline — train CLI
=======================================
Wires the four steps together:
  1. slice (fixed-duration time windows)  -> from irregular/slice.py
  2. extract arrival-pattern features     -> from irregular/features.py
  3. per-machine normalisation            -> shared base.PerMachineNormaliser
  4. fit the chosen model                 -> from irregular/model.py

Saves one fitted pipeline instance as a .pkl bundle containing the
model, normaliser, and feature column names. The reusable part is this
pipeline code; the saved .pkl is machine/data-type specific.

Run from repo root:
    python -m pcil.utils.anomaly.irregular.train \\
        --input data/irregular_dataset.csv \\
        --output data/irregular_inkjet_01.pkl

Models: isolation_forest (default)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pcil.utils.anomaly.base import PerMachineNormaliser
from pcil.utils.anomaly.irregular.slice import detect_windows
from pcil.utils.anomaly.irregular.features import extract_features, stack_features
from pcil.utils.anomaly.irregular.model import IsolationForestModel

_MODEL_REGISTRY = {
    "isolation_forest": IsolationForestModel,
}


def train(
    df: pd.DataFrame,
    model_name: str = "isolation_forest",
    *,
    machine_id_column: str = "machine_id",
    timestamp_column: str = "timestamp",
    value_column: str | None = None,
    window_seconds: float = 1.0,
    model_kwargs: dict | None = None,
) -> dict:
    """
    Run the full pipeline on `df` and return a bundle dict.

    The bundle contains the fitted model, normaliser, and metadata
    needed by score.py to score new data. Train on NORMAL-operation
    data only — the model learns the baseline arrival pattern, and the
    threshold marks departures from it.
    """
    # 1. Slice into fixed-duration windows, extract features per window
    window_rows = []
    for machine_id, group in df.groupby(machine_id_column):
        group = group.sort_values(timestamp_column).reset_index(drop=True)
        for _w_start, start, end in detect_windows(
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
            window_rows.append(features)

    feature_df = stack_features(window_rows)
    feature_columns = [c for c in feature_df.columns if c != machine_id_column]

    # 2. Per-machine z-score normalisation
    normaliser = PerMachineNormaliser()
    feature_df_norm = normaliser.fit_transform(
        feature_df,
        machine_id_column=machine_id_column,
        feature_columns=feature_columns,
    )

    # 3. Fit the chosen model
    model_cls = _MODEL_REGISTRY[model_name]
    model = model_cls(**(model_kwargs or {}))
    X = feature_df_norm[feature_columns].to_numpy(dtype=float)
    model.fit(X)

    # Same convention as the cyclical pipeline: without labels at train
    # time, start at the 95th percentile of training scores (top 5% of
    # training windows flagged as suspicious) and refine against
    # labelled data later.
    train_scores = model.score(X)
    best_thresh = float(np.percentile(train_scores, 95))

    return {
        "model":               model,
        "model_name":          model_name,
        "normaliser":          normaliser,
        "feature_columns":     feature_columns,
        "trained_machine_ids": sorted(feature_df[machine_id_column].unique()),
        "machine_id_column":   machine_id_column,
        "timestamp_column":    timestamp_column,
        "value_column":        value_column,
        "window_seconds":      window_seconds,
        "threshold":           best_thresh,
    }


def main():
    parser = argparse.ArgumentParser(description="Train irregular anomaly model.")
    parser.add_argument("--input",  required=True, help="Path to the training CSV (normal operation).")
    parser.add_argument("--output", required=True, help="Where to save the .pkl bundle.")
    parser.add_argument("--model",  choices=list(_MODEL_REGISTRY), default="isolation_forest")
    parser.add_argument("--machine-id-column", default="machine_id")
    parser.add_argument("--timestamp-column",  default="timestamp")
    parser.add_argument("--value-column",      default=None,
                        help="Optional numeric column for value_* features.")
    parser.add_argument("--window-seconds",    type=float, default=1.0)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    bundle = train(
        df,
        model_name=args.model,
        machine_id_column=args.machine_id_column,
        timestamp_column=args.timestamp_column,
        value_column=args.value_column,
        window_seconds=args.window_seconds,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"Saved -> {out_path}")
    print(f"Trained {args.model} on {len(bundle['feature_columns'])} features.")


if __name__ == "__main__":
    main()

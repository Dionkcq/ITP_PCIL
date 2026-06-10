"""
Cyclical anomaly pipeline — train CLI
======================================
Wires the four steps together:
  1. slice (cycle detection)     -> slice.py  (peak / zero_crossing / fixed_period)
  2. extract features per cycle  -> features.py (stats / waveform / fft)
  3. per-machine normalisation   -> PerMachineNormaliser
  4. fit the chosen model        -> model.py  (isolation_forest / autoencoder)

The autoencoder adapts its architecture automatically to whatever feature
method is used — no need to specify input_len manually.

Run from repo root:
    python -m pcil.utils.anomaly.cyclical.train \\
        --input data/cyclical_dataset.csv \\
        --output data/inkjet_cyclical.pkl

Models  : isolation_forest | autoencoder (default: autoencoder)
Slicing : set SLICE_METHOD in slice.py   (default: peak)
Features: set FEATURE_METHOD in features.py (default: waveform)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import joblib
import pandas as pd

from pcil.utils.anomaly.base import PerMachineNormaliser
from pcil.utils.anomaly.cyclical.slice import detect_cycles
from pcil.utils.anomaly.cyclical.features import extract_features, stack_features
from pcil.utils.anomaly.cyclical.model import AutoencoderModel, IsolationForestModel

_MODEL_REGISTRY = {
    "isolation_forest": IsolationForestModel,
    "autoencoder":      AutoencoderModel,
}


def train(
    df: pd.DataFrame,
    model_name: str = "autoencoder",
    *,
    machine_id_column: str = "machine_id",
    signal_column: str = "signal_value",
    timestamp_column: str = "timestamp",
    model_kwargs: dict | None = None,
) -> dict:
    """
    Run the full pipeline on `df` and return a bundle dict.
    input_len is inferred automatically from the feature matrix shape.
    """
    # 1. Slice into cycles, extract features per cycle
    cycle_rows = []
    for machine_id, group in df.groupby(machine_id_column):
        group = group.sort_values(timestamp_column).reset_index(drop=True)
        for start, end in detect_cycles(
            group, signal_column=signal_column, timestamp_column=timestamp_column,
        ):
            features = extract_features(group.iloc[start:end], signal_column=signal_column)
            features[machine_id_column] = machine_id
            cycle_rows.append(features)

    feature_df = stack_features(cycle_rows)
    feature_columns = [c for c in feature_df.columns if c != machine_id_column]

    # 2. Per-machine z-score normalisation
    normaliser = PerMachineNormaliser()
    feature_df_norm = normaliser.fit_transform(
        feature_df,
        machine_id_column=machine_id_column,
        feature_columns=feature_columns,
    )

    # 3. Fit the chosen model — input_len inferred from feature matrix
    model_cls = _MODEL_REGISTRY[model_name]
    model = model_cls(**(model_kwargs or {}))
    X = feature_df_norm[feature_columns].to_numpy(dtype=float)
    model.fit(X)

    # 4. Compute initial threshold (95th percentile of training scores)
    # Refine this by running check_scores.py after scoring labelled eval data
    train_scores = model.score(X)
    threshold = float(np.percentile(train_scores, 95))

    return {
        "model":               model,
        "model_name":          model_name,
        "normaliser":          normaliser,
        "feature_columns":     feature_columns,
        "trained_machine_ids": sorted(feature_df[machine_id_column].unique()),
        "machine_id_column":   machine_id_column,
        "signal_column":       signal_column,
        "timestamp_column":    timestamp_column,
        "threshold":           threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Train cyclical anomaly model.")
    parser.add_argument("--input",  required=True, help="Path to cyclical_dataset.csv.")
    parser.add_argument("--output", required=True, help="Where to save the .pkl bundle.")
    parser.add_argument("--model",  choices=list(_MODEL_REGISTRY), default="autoencoder")
    parser.add_argument("--machine-id-column", default="machine_id")
    parser.add_argument("--signal-column",     default="signal_value")
    parser.add_argument("--timestamp-column",  default="timestamp")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    bundle = train(
        df,
        model_name=args.model,
        machine_id_column=args.machine_id_column,
        signal_column=args.signal_column,
        timestamp_column=args.timestamp_column,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"Saved -> {out_path}")
    print(f"Trained {args.model} on {len(bundle['feature_columns'])} features.")


if __name__ == "__main__":
    main()
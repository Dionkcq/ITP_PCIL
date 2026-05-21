"""
Cyclical anomaly pipeline — train CLI
======================================
Wires the four steps together:
  1. slice (cycle detection)            -> from cyclical/slice.py
  2. extract features per cycle         -> from cyclical/features.py
  3. per-machine normalisation          -> shared normalise.PerMachineNormaliser
  4. fit the chosen model               -> from cyclical/model.py

Saves one fitted pipeline instance as a .pkl bundle containing the
model, normaliser, and feature column names. The reusable part is this
pipeline code; the saved .pkl is machine/data-type specific.

Run from PCIL_dev/:
    python -m pcil.utils.anomaly.cyclical.train \\
        --input ../data/cyclical_dataset.csv \\
        --output ../data/cyclical_inkjet_01.pkl \\
        --model isolation_forest

Models: z_score | isolation_forest | one_class_svm | autoencoder
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from pcil.utils.anomaly.non_cyclical.normalise import PerMachineNormaliser
from pcil.utils.anomaly.cyclical.slice import detect_cycles
from pcil.utils.anomaly.cyclical.features import extract_features, stack_features
from pcil.utils.anomaly.cyclical.model import (
    AutoencoderModel,
    IsolationForestModel,
    OneClassSVMModel,
    ZScoreModel,
)

_MODEL_REGISTRY = {
    "z_score":          ZScoreModel,
    "isolation_forest": IsolationForestModel,
    "one_class_svm":    OneClassSVMModel,
    "autoencoder":      AutoencoderModel,
}


def train(
    df: pd.DataFrame,
    model_name: str,
    *,
    machine_id_column: str = "machine_id",
    signal_column: str = "signal_value",
    timestamp_column: str = "timestamp",
    model_kwargs: dict | None = None,
) -> dict:
    """
    Run the full pipeline on `df` and return a bundle dict that
    score.py / joblib can roundtrip.

    Production intent: run this once per machine/data type. The code can
    carry multiple machine IDs for experiments, but the normaliser/model
    inside the returned bundle are fitted state, not the shared pipeline
    definition itself.
    """
    # 1. Slice into cycles per machine, extract features per cycle
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

    # 3. Fit the chosen model
    model_cls = _MODEL_REGISTRY[model_name]
    model = model_cls(**(model_kwargs or {}))
    X = feature_df_norm[feature_columns].to_numpy(dtype=float)
    model.fit(X)

    return {
        "model":             model,
        "model_name":        model_name,
        "normaliser":        normaliser,
        "feature_columns":   feature_columns,
        "trained_machine_ids": sorted(feature_df[machine_id_column].unique()),
        "machine_id_column": machine_id_column,
        "signal_column":     signal_column,
        "timestamp_column":  timestamp_column,
    }


def main():
    parser = argparse.ArgumentParser(description="Train cyclical anomaly model.")
    parser.add_argument("--input", required=True, help="Path to cyclical_dataset.csv.")
    parser.add_argument("--output", required=True, help="Where to save the bundled .pkl.")
    parser.add_argument("--model", choices=list(_MODEL_REGISTRY), default="isolation_forest")
    parser.add_argument("--machine-id-column", default="machine_id")
    parser.add_argument("--signal-column", default="signal_value")
    parser.add_argument("--timestamp-column", default="timestamp")
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

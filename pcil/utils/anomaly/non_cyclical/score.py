import argparse
from pathlib import Path

import joblib
import pandas as pd

from pcil.utils.anomaly.non_cyclical.slice import detect_windows
from pcil.utils.anomaly.non_cyclical.features import extract_features, stack_features


def load_acoustic(csv_path: Path, *, header_skiprows: int = 5) -> pd.DataFrame:
    """Load an acoustic CSV, skipping its metadata header lines."""
    return pd.read_csv(csv_path, skiprows=header_skiprows)


def score(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Return a per-window DataFrame with features + anomaly_score."""
    machine_id        = bundle["machine_id"]
    machine_id_column = bundle["machine_id_column"]
    window_size_rows  = bundle["window_size_rows"]
    channel_columns   = bundle["channel_columns"]
    feature_columns   = bundle["feature_columns"]
    normaliser        = bundle["normaliser"]
    model             = bundle["model"]

    df_sorted = df.reset_index(drop=True)
    rows = []
    for start, end in detect_windows(df_sorted, window_size_rows=window_size_rows):
        features = extract_features(df_sorted.iloc[start:end], channel_columns=channel_columns)
        features[machine_id_column] = machine_id
        features["window_start_idx"] = start
        rows.append(features)

    if not rows:
        # Input shorter than one window (window_size_rows) -> no windows.
        # Return an empty result with the expected columns so callers see
        # "0 windows scored", not an error. (Checked before stack_features,
        # which raises ValueError on an empty list.)
        return pd.DataFrame(
            columns=[*feature_columns, machine_id_column,
                     "window_start_idx", "anomaly_score"]
        )

    feature_df      = stack_features(rows)
    feature_df_norm = normaliser.transform(feature_df, machine_id_column=machine_id_column)
    X               = feature_df_norm[feature_columns].to_numpy(dtype=float)
    feature_df["anomaly_score"] = model.score(X)
    return feature_df


def main():
    parser = argparse.ArgumentParser(description="Score non-cyclical data with a trained model.")
    parser.add_argument("--input",           required=True, help="Acoustic CSV to score.")
    parser.add_argument("--model",           required=True, help="Path to .pkl from run.py.")
    parser.add_argument("--output",          required=True, help="Where to save the scored CSV.")
    parser.add_argument("--header-skiprows", type=int, default=5)
    args = parser.parse_args()

    df     = load_acoustic(Path(args.input), header_skiprows=args.header_skiprows)
    bundle = joblib.load(args.model)
    scored = score(df, bundle)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    print(f"Scored {len(scored)} windows.")


if __name__ == "__main__":
    main()
"""
PCIL Pipeline #1 — shop-floor DataFrame slice -> Golden DataFrame
=================================================================
Reads a slice pulled from the shop-floor database (provided here as a
CSV path) and produces the Golden DataFrame Pipeline #2 consumes.

    shop-floor DB -> trigger slice -> preprocess.py -> Golden DataFrame
                                      (this script)         (output)

The transformation is built around sklearn.pipeline.Pipeline +
ColumnTransformer:
  - numerical features  -> MinMaxScaler           (clamps each to [0, 1])
  - categorical features -> OneHotEncoder         (sparse_output=False)
  - timestamp + targets  -> passed through unchanged

Run from PCIL_dev/:
    python pcil/preprocess.py --input path/to/slice.csv
    python pcil/preprocess.py --input slice.csv --config path/to/config.yaml
    python pcil/preprocess.py --input slice.csv --config inkjet_printer

The input CSV must satisfy the schema declared in config.yaml's `input` block.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Note: sys.stdout reassignment for UTF-8 lives inside main() now (not at
# module level) so importing this module doesn't break pytest's stdout
# capture — same pattern train_context_model.py uses.


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

def resolve_config_path(arg: str | None = None) -> Path:
    """Resolve a CLI arg into a config.yaml path. Defaults to inkjet_printer."""
    repo_root = Path(__file__).resolve().parent.parent  # PCIL_dev/
    if arg:
        p = Path(arg)
        if p.is_file():
            return p.resolve()
        return repo_root / "machines" / arg / "config.yaml"
    return repo_root / "machines" / "inkjet_printer" / "config.yaml"


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    config_dir = config_path.parent
    output_dir = (config_dir / cfg["pipeline"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg["_paths"] = {
        "config_dir": config_dir,
        "output":     output_dir,
    }
    return cfg


# ─────────────────────────────────────────────────────────────
# Pipeline construction
# ─────────────────────────────────────────────────────────────

def build_preprocessor(numerical: list[str], categorical: list[str]) -> Pipeline:
    """
    Build sklearn Pipeline wrapping a ColumnTransformer:
        numerical   -> MinMaxScaler        (each feature -> [0, 1])
        categorical -> OneHotEncoder       (sparse_output=False)
    Other columns are dropped (timestamp + targets are reattached separately).
    """
    transformers = []
    if numerical:
        transformers.append(("num", MinMaxScaler(), numerical))
    if categorical:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical,
        ))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline([("transform", column_transformer)])


def feature_names_out(pipeline: Pipeline) -> list[str]:
    """Names of the feature columns the pipeline emits (post-fit)."""
    transformer = pipeline.named_steps["transform"]
    return list(transformer.get_feature_names_out())


# ─────────────────────────────────────────────────────────────
# Core preprocessing
# ─────────────────────────────────────────────────────────────

def preprocess(input_df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, Pipeline]:
    """
    Convert a shop-floor DataFrame slice into the Golden DataFrame.

    Steps:
      1. Drop legacy `scenario` column if present.
      2. Validate required columns are on the input.
      3. Fit + apply ColumnTransformer.
      4. Assemble output: timestamp + targets + transformed features.

    Returns (golden_df, fitted_pipeline). The fitted pipeline is returned
    so callers can persist it alongside the trained model.
    """
    schema = cfg["input"]
    timestamp_col = schema["timestamp_column"]
    numerical = list(schema.get("numerical_features", []) or [])
    categorical = list(schema.get("categorical_features", []) or [])
    targets = list(schema.get("targets", []) or [])

    # 1. Drop the legacy scenario column. Week-1 carried it as the per-row
    # context; the Week-2 architecture treats trigger time-windows as
    # context, so this column is no longer part of the contract.
    df = input_df.drop(columns=["scenario"], errors="ignore")

    # 2. Schema validation
    required = {timestamp_col, *numerical, *categorical, *targets}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"preprocess: input slice is missing columns: {sorted(missing)}"
        )

    # 3. Fit + apply ColumnTransformer
    pipeline = build_preprocessor(numerical, categorical)
    transformed = pipeline.fit_transform(df)
    feat_names = feature_names_out(pipeline)
    transformed_df = pd.DataFrame(transformed, columns=feat_names, index=df.index)

    # 4. Reassemble Golden DataFrame: timestamp -> targets -> features
    golden = pd.DataFrame()
    golden[timestamp_col] = df[timestamp_col].values
    for t in targets:
        golden[t] = df[t].values
    for f in feat_names:
        golden[f] = transformed_df[f].values

    return golden, pipeline


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────

def print_summary(golden: pd.DataFrame, cfg: dict) -> None:
    timestamp_col = cfg["input"]["timestamp_column"]
    targets = cfg["input"]["targets"]
    feat_cols = [c for c in golden.columns if c not in targets and c != timestamp_col]

    print("\n" + "=" * 70)
    print("GOLDEN DATAFRAME (Pipeline #1, refactored)")
    print("=" * 70)
    print(f"Shape: {golden.shape[0]} rows x {golden.shape[1]} cols")
    print(f"Columns: {list(golden.columns)}")
    print()
    print("Feature value ranges (should sit in [0, 1] after MinMaxScaler):")
    print(golden[feat_cols].agg(["min", "max", "mean"]).round(4).to_string())
    print()
    print("Target columns (passed through):")
    print(golden[targets].agg(["min", "max", "mean"]).round(4).to_string())
    print("=" * 70)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="PCIL Pipeline #1 — shop-floor slice -> Golden DataFrame.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python pcil/preprocess.py --input slice.csv\n"
            "  python pcil/preprocess.py --input slice.csv --config inkjet_printer\n"
            "  python pcil/preprocess.py --input slice.csv --config path/to/config.yaml\n"
        ),
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a CSV with a shop-floor DataFrame slice.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config.yaml or a machine name. "
             "Defaults to machines/inkjet_printer/config.yaml.",
    )
    parser.add_argument(
        "--save-pipeline", action="store_true",
        help="Persist the fitted ColumnTransformer to output/preprocessor.pkl.",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    if not config_path.is_file():
        print(f"Config not found: {config_path}")
        raise SystemExit(1)

    print(f"[1/3] Loading config: {config_path}")
    cfg = load_config(config_path)

    print(f"[2/3] Loading input slice: {args.input}")
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        print(f"Input file not found: {input_path}")
        raise SystemExit(1)
    input_df = pd.read_csv(input_path)
    print(f"      Input shape: {input_df.shape}")

    print(f"[3/3] Preprocessing -> Golden DataFrame...")
    golden, pipeline = preprocess(input_df, cfg)

    out_dir = cfg["_paths"]["output"]
    out_path = out_dir / "golden_dataframe.csv"
    golden.to_csv(out_path, index=False)
    print(f"      Saved -> {out_path}")

    if args.save_pipeline:
        pkl_path = out_dir / "preprocessor.pkl"
        joblib.dump(pipeline, pkl_path)
        print(f"      Saved -> {pkl_path}")

    print_summary(golden, cfg)


if __name__ == "__main__":
    main()

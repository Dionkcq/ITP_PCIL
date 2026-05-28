"""
Pipeline #1 -> Pipeline #2 Adapter (stub)
=========================================
Translates the Golden DataFrame (pandas, labelled, mixed types) into
the plain numpy arrays the Context Model wants.

Three chores, in order. Nothing else.
  1. Validate every requested column is present.
  2. Validate feature values are in [0, 1] (Pipeline #1's contract).
  3. Reshape into separate X (features) and y (targets) arrays.

Pure function: same input -> same output, no side effects, no I/O,
no imputation, no business logic. If the input is wrong it raises —
Pipeline #1 is responsible for fixing the cause, not us.

Usage:
    from adapter import adapt, column_names_from_config

    cfg    = yaml.safe_load(open("machines/inkjet_printer/config.yaml"))
    df     = pd.read_csv("machines/inkjet_printer/output/golden_dataframe.csv")
    targets, features = column_names_from_config(cfg, df)
    bundle = adapt(df, targets, features)

CLI demo (run from PCIL_dev/):
    python pcil/adapter.py                # default: inkjet_printer
    python pcil/adapter.py oil_filler     # by machine name
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def adapt(
    golden_df: pd.DataFrame,
    target_cols: list[str],
    feature_cols: list[str],
) -> dict:
    """
    Convert a Golden DataFrame into Context-Model-ready arrays.

    Returns a dict:
        X:             (n_rows, n_features) numpy array
        y:             (n_rows, n_targets)  numpy array
        feature_names: list[str]
        target_names:  list[str]
        n_rows:        int

    Raises ValueError on missing columns, NaN values, or feature
    values outside [0, 1].
    """
    # 1. Schema check
    requested = set(target_cols) | set(feature_cols)
    missing   = requested - set(golden_df.columns)
    if missing:
        raise ValueError(
            f"adapter: golden_df is missing required columns: {sorted(missing)}"
        )

    feats = golden_df[feature_cols]

    # 2a. NaN check
    if feats.isna().any().any():
        bad = feats.columns[feats.isna().any()].tolist()
        raise ValueError(f"adapter: feature columns contain NaN: {bad}")

    # 2b. Range check. Tolerate ~1e-9 floating-point overshoot from
    # MinMaxScaler — on small inputs the scaler can produce values like
    # 1.0000000000000002 that should logically be 1.0.
    EPS = 1e-9
    fmin = float(feats.min().min())
    fmax = float(feats.max().max())
    if fmin < -EPS or fmax > 1.0 + EPS:
        raise ValueError(
            f"adapter: feature values outside [0, 1] "
            f"(min={fmin:.4f}, max={fmax:.4f})"
        )

    # 3. Strip labels and reshape
    X = feats.to_numpy(dtype=float)
    y = golden_df[target_cols].to_numpy(dtype=float)

    return {
        "X":             X,
        "y":             y,
        "feature_names": list(feature_cols),
        "target_names":  list(target_cols),
        "n_rows":        int(X.shape[0]),
    }


def column_names_from_config(
    cfg: dict,
    golden_df: pd.DataFrame | None = None,
) -> tuple[list[str], list[str]]:
    """
    Pull (targets, features) for the adapter from a config.yaml.

    Targets come from cfg["input"]["targets"] (declarative).

    Features depend on what ColumnTransformer produced:
      - If golden_df is provided, derive features from the DataFrame
        columns (everything that isn't a target or the timestamp).
        This is the most robust path because it correctly captures
        post-OneHotEncoder column expansion.
      - Otherwise return the INPUT-level numerical + categorical names
        from the YAML. Use only when no categorical features exist.
    """
    targets = list(cfg["input"]["targets"])

    if golden_df is not None:
        timestamp_col = cfg["input"]["timestamp_column"]
        features = [
            c for c in golden_df.columns
            if c not in targets and c != timestamp_col
        ]
        return targets, features

    features = (
        list(cfg["input"].get("numerical_features", []) or [])
        + list(cfg["input"].get("categorical_features", []) or [])
    )
    return targets, features


# ─────────────────────────────────────────────────────────────────────
# Demo / smoke test
# ─────────────────────────────────────────────────────────────────────

def _resolve_machine(arg: str | None) -> Path:
    """Resolve a CLI arg → config.yaml path."""
    repo_root = Path(__file__).resolve().parent.parent  # PCIL_dev/
    if arg:
        p = Path(arg)
        if p.is_file():
            return p.resolve()
        return repo_root / "machines" / arg / "config.yaml"
    return repo_root / "machines" / "inkjet_printer" / "config.yaml"


if __name__ == "__main__":
    import yaml

    # UTF-8 stdout for Windows — only when run as a script
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    cfg_path = _resolve_machine(arg)
    if not cfg_path.is_file():
        print(f"[demo] Config not found: {cfg_path}")
        raise SystemExit(1)

    machine_dir = cfg_path.parent
    output_dir  = machine_dir / yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["pipeline"]["output_dir"]
    csv_path    = output_dir / "golden_dataframe.csv"

    if not csv_path.exists():
        print(f"[demo] Golden DataFrame not found at {csv_path}")
        print(f"[demo] Run preprocess.py first to generate it.")
        raise SystemExit(1)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    df  = pd.read_csv(csv_path)

    targets, features = column_names_from_config(cfg, df)
    bundle = adapt(df, targets, features)

    print("-" * 60)
    print("ADAPTER STUB - demo run")
    print("-" * 60)
    print(f"Machine (config folder): {machine_dir.name}")
    print(f"Input  Golden DataFrame: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Output X (features):     shape {bundle['X'].shape}")
    print(f"Output y (targets):      shape {bundle['y'].shape}")
    print(f"Feature names: {bundle['feature_names']}")
    print(f"Target  names: {bundle['target_names']}")
    print()
    print("First row of X (features):")
    print(f"  {bundle['X'][0]}")
    print("First row of y (targets):")
    print(f"  {bundle['y'][0]}")
    print("-" * 60)
    print("OK - Pipeline #2 can consume this.")

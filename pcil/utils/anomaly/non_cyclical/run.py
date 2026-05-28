"""
Non-cyclical anomaly training CLI.
====================================
Zi Hin's original training + evaluation workflow, preserved as a thin
CLI wrapper around the importable function in train.py.

This script loads recordings from disk using paths in
`non_cyclical_config.yaml`, calls the shared `train_from_clean_and_anomaly`
function, saves the bundle, then runs Zi Hin's precision/recall threshold
sweep on held-out test data. The CLI behaviour is identical to before
the refactor.

Run from PCIL_dev/ (or PCIL/):
    python pcil/utils/anomaly/non_cyclical/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

# Two roots — kept separate because they aren't always the same directory.
#   PACKAGE_ROOT: parent of pcil/. Added to sys.path so `from pcil.xxx` resolves.
#   ROOT_DIR:     where the data/ folder lives. parents[3] on Zi Hin's PCIL/ clone;
#                 parents[4] on the PCIL_dev/ sandbox. Pick whichever has data/.
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))

_data_candidates = [PACKAGE_ROOT, PACKAGE_ROOT.parent]
ROOT_DIR = next(
    (p for p in _data_candidates if (p / "data").is_dir()),
    PACKAGE_ROOT,
)

from pcil.utils.anomaly.non_cyclical.score import load_acoustic, score
from pcil.utils.anomaly.non_cyclical.train import train_from_clean_and_anomaly


def main() -> None:
    with open(SCRIPT_DIR / "non_cyclical_config.yaml") as f:
        cfg = yaml.safe_load(f)

    input_clean           = ROOT_DIR / cfg["data"]["input_clean"]
    input_anomaly         = ROOT_DIR / cfg["data"]["input_anomaly"]
    output_model          = ROOT_DIR / cfg["data"]["output_model"]
    output_clean_scored   = ROOT_DIR / cfg["data"]["output_clean_scored"]
    output_anomaly_scored = ROOT_DIR / cfg["data"]["output_anomaly_scored"]

    machine_id       = cfg["machine"]["machine_id"]
    header_skiprows  = cfg["machine"]["header_skiprows"]
    window_size_rows = cfg["model"]["window_size_rows"]
    train_ratio      = cfg["model"]["train_ratio"]

    # ── Step 0: Load + split ─────────────────────────────────────
    print("=" * 60)
    print("STEP 0 — Loading data and splitting 80/20")
    print("=" * 60)
    df_clean   = load_acoustic(input_clean,   header_skiprows=header_skiprows)
    df_anomaly = load_acoustic(input_anomaly, header_skiprows=header_skiprows)

    n_clean_train   = int(len(df_clean)   * train_ratio)
    n_anomaly_train = int(len(df_anomaly) * train_ratio)
    print(f"Clean    — total: {len(df_clean)} rows  train: {n_clean_train}  test: {len(df_clean) - n_clean_train}")
    print(f"Anomaly  — total: {len(df_anomaly)} rows  train: {n_anomaly_train}  test: {len(df_anomaly) - n_anomaly_train}")

    # ── Steps 1–3: Extract features, normalise, train ────────────
    print("=" * 60)
    print("STEP 1-3 — Extracting features, per-machine normalisation, training Random Forest")
    print("=" * 60)
    bundle = train_from_clean_and_anomaly(
        df_clean, df_anomaly,
        machine_id=machine_id,
        window_size_rows=window_size_rows,
        train_ratio=train_ratio,
    )
    print(f"  trained on clean windows:   {bundle['trained_window_counts']['clean']}")
    print(f"  trained on anomaly windows: {bundle['trained_window_counts']['anomaly']}")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_model)
    print(f"Bundle saved -> {output_model}")

    # ── Step 4: Score the held-out test split via score.score() ─
    print("=" * 60)
    print("STEP 4 — Scoring test split (unseen data) via score.score()")
    print("=" * 60)
    clean_test   = df_clean.iloc[n_clean_train:]
    anomaly_test = df_anomaly.iloc[n_anomaly_train:]

    scored_clean_test   = score(clean_test,   bundle)
    scored_anomaly_test = score(anomaly_test, bundle)

    scored_clean_test[["window_start_idx", "anomaly_score"]].to_csv(
        output_clean_scored, index=False)
    scored_anomaly_test[["window_start_idx", "anomaly_score"]].to_csv(
        output_anomaly_scored, index=False)
    print(f"Saved -> {output_clean_scored}")
    print(f"Saved -> {output_anomaly_scored}")

    # ── Step 5: Threshold sweep on held-out data ────────────────
    print("=" * 60)
    print("STEP 5 — Evaluating on unseen test data")
    print("=" * 60)
    scores_clean_test   = scored_clean_test["anomaly_score"].to_numpy()
    scores_anomaly_test = scored_anomaly_test["anomaly_score"].to_numpy()

    print("\n=== Threshold sweep (test split only) ===")
    for q in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]:
        threshold = pd.Series(scores_clean_test).quantile(q)
        tp = (scores_anomaly_test > threshold).sum()
        fp = (scores_clean_test   > threshold).sum()
        fn = (scores_anomaly_test <= threshold).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"q={q:.2f}  threshold={threshold:.4f}  Precision={precision:.3f}  "
              f"Recall={recall:.3f}  TP={tp} FP={fp} FN={fn}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

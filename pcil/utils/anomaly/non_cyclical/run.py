from pathlib import Path
import sys
import numpy as np
import joblib
import pandas as pd
import yaml

# Two roots — kept separate because they aren't always the same directory.
#
#   PACKAGE_ROOT: parent of pcil/. Always parents[3] of this script. Added to
#                 sys.path so absolute imports like `from pcil.xxx` resolve.
#
#   ROOT_DIR (data root): where the `data/` folder lives. On Zi Hin's PCIL/
#                 clone this is also parents[3]; on our PCIL_dev/ sandbox it
#                 is parents[4] (one level higher, in ITP/). Pick the first
#                 candidate that actually has data/ on disk.
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))

_data_candidates = [PACKAGE_ROOT, PACKAGE_ROOT.parent]
ROOT_DIR = next(
    (p for p in _data_candidates if (p / "data").is_dir()),
    PACKAGE_ROOT,
)

from pcil.utils.anomaly.non_cyclical.score import load_acoustic, score
from pcil.utils.anomaly.non_cyclical.features import extract_features, stack_features, DEFAULT_FEATURE_NAMES, CHANNEL_COLUMNS
from pcil.utils.anomaly.non_cyclical.slice import detect_windows
from pcil.utils.anomaly.base import PerMachineNormaliser
from pcil.utils.anomaly.non_cyclical.model import RandomForestModel

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

# ── Helper: extract features from a dataframe ─────────────────────────────────
def extract_all_features(df):
    rows = []
    for start, end in detect_windows(df, window_size_rows=window_size_rows):
        feats = extract_features(df.iloc[start:end],
                                 channel_columns=CHANNEL_COLUMNS,
                                 feature_names=DEFAULT_FEATURE_NAMES)
        feats["machine_id"] = machine_id
        rows.append(feats)
    return stack_features(rows)

# ── Step 0: Load and split ────────────────────────────────────────────────────
print("=" * 60)
print("STEP 0 — Loading data and splitting 80/20")
print("=" * 60)

df_clean   = load_acoustic(input_clean,   header_skiprows=header_skiprows)
df_anomaly = load_acoustic(input_anomaly, header_skiprows=header_skiprows)

# Time-series split — no shuffle, order must be preserved
clean_train   = df_clean.iloc[:int(len(df_clean)   * train_ratio)]
clean_test    = df_clean.iloc[int(len(df_clean)    * train_ratio):]
anomaly_train = df_anomaly.iloc[:int(len(df_anomaly) * train_ratio)]
anomaly_test  = df_anomaly.iloc[int(len(df_anomaly)  * train_ratio):]

print(f"Clean    — train: {len(clean_train)} rows  test: {len(clean_test)} rows")
print(f"Anomaly  — train: {len(anomaly_train)} rows  test: {len(anomaly_test)} rows")

# ── Step 1: Extract features ──────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Extracting features")
print("=" * 60)

feat_clean_train   = extract_all_features(clean_train)
feat_clean_test    = extract_all_features(clean_test)
feat_anomaly_train = extract_all_features(anomaly_train)
feat_anomaly_test  = extract_all_features(anomaly_test)

print(f"Train windows — clean: {len(feat_clean_train)}  anomaly: {len(feat_anomaly_train)}")
print(f"Test windows  — clean: {len(feat_clean_test)}   anomaly: {len(feat_anomaly_test)}")

# ── Step 2: Normalise ─────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2 — Per-machine normalisation")
print("=" * 60)

feature_cols = [c for c in feat_clean_train.columns if c != "machine_id"]

# Fit ONLY on clean training data — never on test or anomaly data
normaliser = PerMachineNormaliser()
feat_clean_train_n   = normaliser.fit_transform(feat_clean_train, machine_id_column="machine_id", feature_columns=feature_cols)
feat_clean_test_n    = normaliser.transform(feat_clean_test,    machine_id_column="machine_id")
feat_anomaly_train_n = normaliser.transform(feat_anomaly_train, machine_id_column="machine_id")
feat_anomaly_test_n  = normaliser.transform(feat_anomaly_test,  machine_id_column="machine_id")

print("Normaliser fitted on clean training data only.")

# ── Step 3: Train ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3 — Training Random Forest on train split")
print("=" * 60)

X_train = pd.concat([feat_clean_train_n[feature_cols], feat_anomaly_train_n[feature_cols]], ignore_index=True).to_numpy()
y_train = np.array([0] * len(feat_clean_train_n) + [1] * len(feat_anomaly_train_n))

model = RandomForestModel()
model.fit(X_train, y_train)

bundle = {
    "model":              model,
    "normaliser":         normaliser,
    "feature_columns":    feature_cols,
    "machine_id":         machine_id,
    "machine_id_column":  "machine_id",
    "window_size_rows":   window_size_rows,
    "channel_columns":    CHANNEL_COLUMNS,
}
joblib.dump(bundle, output_model)
print(f"Model saved -> {output_model}")

# ── Step 4: Score test split ──────────────────────────────────────────────────
# Use the public score() function from score.py — same path the orchestrator
# (and engineer-facing API) calls. Verifies the production scoring path end
# to end on labelled test data.
print("=" * 60)
print("STEP 4 — Scoring test split (unseen data) via score.score()")
print("=" * 60)

scored_clean_test   = score(clean_test,   bundle)
scored_anomaly_test = score(anomaly_test, bundle)

scores_clean_test   = scored_clean_test["anomaly_score"].to_numpy()
scores_anomaly_test = scored_anomaly_test["anomaly_score"].to_numpy()

scored_clean_test[["window_start_idx", "anomaly_score"]].to_csv(output_clean_scored,   index=False)
scored_anomaly_test[["window_start_idx", "anomaly_score"]].to_csv(output_anomaly_scored, index=False)
print(f"Saved -> {output_clean_scored}")
print(f"Saved -> {output_anomaly_scored}")

# ── Step 5: Evaluate on test split ────────────────────────────────────────────
print("=" * 60)
print("STEP 5 — Evaluating on unseen test data")
print("=" * 60)

print("\n=== Threshold sweep (test split only) ===")
for q in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]:
    threshold = pd.Series(scores_clean_test).quantile(q)
    tp = (scores_anomaly_test > threshold).sum()
    fp = (scores_clean_test   > threshold).sum()
    fn = (scores_anomaly_test <= threshold).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"q={q:.2f}  threshold={threshold:.4f}  Precision={precision:.3f}  Recall={recall:.3f}  TP={tp} FP={fp} FN={fn}")

print("\n" + "=" * 60)
print("Pipeline complete.")
print("=" * 60)
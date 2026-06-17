"""
Pipeline #2 — Context Model (Linear Regression baseline)
========================================================
Two entry points:

  1. `train_context_model_from_df(golden_df, cfg)` — importable function.
     Used by the orchestrator. Returns (impacts_dict, fitted_model).

  2. CLI (this file as a script) — wraps the function for one-off runs.
     Reads the Golden DataFrame from disk, fits the model, saves both
     `context_model.pkl` and `context_model_impacts.json`.

The impacts JSON follows the Week-3 schema agreed with Winardi on
2026-05-22 (`deliverables/Week3/todo.md` §1.3.1):
    system / model / fitted_at
    context_window: { start_time, end_time, row_count, feature_count, target_count }
    context: [ { target, intercept, ranked_feature_impacts: [...] } ]

Each entry in `ranked_feature_impacts` carries: feature name, a one-line
description (pulled from `config.yaml -> feature_descriptions`), the raw
coefficient, a standardised share-of-explanation, and an absolute-magnitude
rank within the target.

Run from PCIL_dev/:
    python -m pcil.train_context_model                # default: inkjet_printer
    python -m pcil.train_context_model oil_filler     # by system name
"""

from __future__ import annotations

# Allow `python pcil/train_context_model.py` AND `python -m pcil.train_context_model`.
# When invoked as a plain script there is no parent package, so `from pcil.xxx`
# imports fail. Promote ourselves into the `pcil` package in that case.
if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "pcil"

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

from pcil.adapter import adapt, column_names_from_config


# ─────────────────────────────────────────────────────────────
# Core function (orchestrator + CLI both call this)
# ─────────────────────────────────────────────────────────────

def train_context_model_from_df(
    golden_df: pd.DataFrame,
    cfg: dict,
) -> tuple[dict, LinearRegression]:
    """Fit a multi-target LinearRegression on a Golden DataFrame.

    Returns
    -------
    (impacts, model)
        impacts : dict matching the Week-3 schema (system, model, fitted_at,
                  context_window, context[].ranked_feature_impacts[]).
        model   : the fitted sklearn LinearRegression.
    """
    targets, features = column_names_from_config(cfg, golden_df)
    bundle = adapt(golden_df, targets, features)
    X, y = bundle["X"], bundle["y"]

    model = LinearRegression().fit(X, y)

    timestamp_col = cfg["input"]["timestamp_column"]
    timestamps = pd.to_datetime(golden_df[timestamp_col])

    feature_descriptions = cfg.get("feature_descriptions", {}) or {}
    system_name = cfg.get("system") or cfg.get("machine") or "unknown_system"

    context_blocks = []
    for i, target_name in enumerate(targets):
        raw = {feat: float(coef) for feat, coef in zip(features, model.coef_[i])}
        sum_abs = sum(abs(c) for c in raw.values()) or 1.0

        # Rank by absolute magnitude (most-impactful feature is rank 1).
        ranked = sorted(raw.items(), key=lambda kv: abs(kv[1]), reverse=True)

        ranked_impacts = [
            {
                "feature": feat,
                "description": feature_descriptions.get(feat, ""),
                "raw_impact_score": coef,
                "standardized_impact_score": coef / sum_abs,
                "rank": rank,
            }
            for rank, (feat, coef) in enumerate(ranked, start=1)
        ]

        context_blocks.append({
            "target": target_name,
            "intercept": float(model.intercept_[i]),
            "ranked_feature_impacts": ranked_impacts,
        })

    impacts = {
        "system": system_name,
        "model": "linear_regression",
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "context_window": {
            "start_time": timestamps.min().isoformat(),
            "end_time": timestamps.max().isoformat(),
            "row_count": int(bundle["n_rows"]),
            "feature_count": int(len(features)),
            "target_count": int(len(targets)),
        },
        "context": context_blocks,
    }

    return impacts, model


def save_artifacts(
    impacts: dict,
    model: LinearRegression,
    output_dir: Path,
    feature_names: list[str],
    target_names: list[str],
) -> tuple[Path, Path]:
    """Persist impacts JSON + fitted model to disk. Returns the two paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "context_model_impacts.json"
    pkl_path = output_dir / "context_model.pkl"
    json_path.write_text(json.dumps(impacts, indent=2), encoding="utf-8")
    joblib.dump({
        "model": model,
        "feature_names": feature_names,
        "target_names": target_names,
    }, pkl_path)
    return json_path, pkl_path


# ─────────────────────────────────────────────────────────────
# CLI wrapper
# ─────────────────────────────────────────────────────────────

def _resolve_system(arg: str | None) -> Path:
    repo_root = Path(__file__).resolve().parent.parent  # PCIL_dev/
    if arg:
        p = Path(arg)
        if p.is_file():
            return p.resolve()
        return repo_root / "systems" / arg / "config.yaml"
    return repo_root / "systems" / "inkjet_printer" / "config.yaml"


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    cfg_path = _resolve_system(arg)
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}")
        raise SystemExit(1)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    system_dir = cfg_path.parent
    output_dir = (system_dir / cfg["pipeline"]["output_dir"]).resolve()

    csv_path = output_dir / "golden_dataframe.csv"
    if not csv_path.exists():
        print(f"Golden DataFrame not found at {csv_path}")
        print(f"Run preprocess.py first.")
        raise SystemExit(1)

    golden_df = pd.read_csv(csv_path)
    impacts, model = train_context_model_from_df(golden_df, cfg)

    targets, features = column_names_from_config(cfg, golden_df)
    json_path, pkl_path = save_artifacts(
        impacts, model, output_dir,
        feature_names=features,
        target_names=targets,
    )

    print(f"Wrote {json_path}")
    print(f"Wrote {pkl_path}")
    print()
    print(f"System:  {impacts['system']}")
    print(f"Model:   {impacts['model']}")
    print(f"Window:  {impacts['context_window']['row_count']} rows "
          f"({impacts['context_window']['start_time']} -> "
          f"{impacts['context_window']['end_time']})")
    print()
    for block in impacts["context"]:
        print(f"  Target: {block['target']}   (intercept {block['intercept']:+.4f})")
        for fi in block["ranked_feature_impacts"]:
            print(f"    [{fi['rank']}] {fi['feature']:<28s} "
                  f"raw={fi['raw_impact_score']:+.4f}  "
                  f"std={fi['standardized_impact_score']:+.4f}")
        print()


if __name__ == "__main__":
    main()

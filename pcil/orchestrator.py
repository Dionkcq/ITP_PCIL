"""
PCIL Job Orchestrator (FastAPI)
================================
The single service that coordinates the runtime pipeline. Replaces the
Week-2 CLI chain (trigger.py -> preprocess.py -> train_context_model.py)
with a set of HTTP endpoints, per Winardi's 22 May 2026 redesign.

Two endpoint groups:

  /pipeline/*    The main runtime path.
    POST /pipeline/run        — full pipeline: pull slice -> preprocess
                                 -> adapter -> context model -> impacts.
                                 RAG + LLM are placeholders for now.
    POST /pipeline/save_csv   — pull a slice and write
                                 `context_window_<start>_<end>.csv`.
                                 Optional; not called during /run.

  /anomaly/*     Engineer-facing API for anomaly scoring.
    POST /anomaly/score       — input: time-series data + model_type.
                                 Output: anomaly score.
                                 STUB: teammates' models still in progress.

Run from PCIL_dev/:
    pip install -r requirements.txt
    uvicorn pcil.orchestrator:app --reload --host 0.0.0.0 --port 8000

OpenAPI / Swagger UI:  http://localhost:8000/docs
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pcil.adapter import adapt, column_names_from_config
from pcil.preprocess import load_config, preprocess
from pcil.train_context_model import save_artifacts, train_context_model_from_df
from pcil.trigger import slice_by_time, slice_last_n_rows
from pcil.utils.anomaly.cyclical.score import score as cyclical_score
from pcil.utils.anomaly.non_cyclical.score import score as non_cyclical_score

# Project root for resolving relative data paths (e.g. anomaly bundles).
#   - In dev (PCIL_dev/): orchestrator.py -> pcil/ -> PCIL_dev/ -> ITP/ is the
#     project root, so parents[2] is correct.
#   - In Docker (/app/): orchestrator.py -> pcil/ -> /app/ is the root, so
#     parents[2] would be /, which is wrong. Set PCIL_PROJECT_ROOT=/app in
#     the Dockerfile to override.
PROJECT_ROOT = Path(
    os.environ.get("PCIL_PROJECT_ROOT") or Path(__file__).resolve().parents[2]
)


app = FastAPI(
    title="PCIL Job Orchestrator",
    version="0.1.0",
    description=(
        "Coordinates the PCIL runtime pipeline. See /docs for the "
        "interactive Swagger UI."
    ),
)


# ─────────────────────────────────────────────────────────────
# Pydantic request / response schemas
# ─────────────────────────────────────────────────────────────

class RunPipelineRequest(BaseModel):
    config_path: str = Field(
        ...,
        description=(
            "Path to a config.yaml recipe. Relative paths resolve from "
            "the orchestrator's working directory (typically PCIL_dev/)."
        ),
        examples=["machines/inkjet_printer/config.yaml"],
    )
    persist: bool = Field(
        False,
        description=(
            "If true, also save context_model.pkl + impacts JSON to the "
            "output directory specified in config.yaml. Defaults to false "
            "so /run can be called repeatedly without overwriting artifacts."
        ),
    )


class SaveCsvRequest(BaseModel):
    config_path: str = Field(..., examples=["machines/inkjet_printer/config.yaml"])


class AnomalyScoreRequest(BaseModel):
    data: list[dict[str, Any]] = Field(
        ...,
        description="Time-series rows. Schema depends on the anomaly model.",
    )
    model_type: Literal["cyclical", "non_cyclical"] = Field(
        ...,
        description="Which anomaly model to invoke.",
    )
    model_id: str | None = Field(
        None,
        description=(
            "Optional identifier when multiple trained instances exist for "
            "the same model_type (e.g. per machine)."
        ),
    )


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _resolve_config(config_path: str) -> tuple[Path, dict]:
    """Resolve + load a config.yaml. Raises HTTPException on failure."""
    p = Path(config_path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if not p.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"config.yaml not found at {p}",
        )
    try:
        cfg = load_config(p)
    except Exception as exc:  # noqa: BLE001 — surface any YAML / IO error
        raise HTTPException(status_code=400, detail=f"failed to load config: {exc}") from exc
    return p, cfg


def _pull_slice(cfg: dict) -> pd.DataFrame:
    """Pull the shop-floor slice described by cfg['trigger'].

    For now the source is a CSV path (mock shop-floor). When the
    engineering team's real DB comes online, replace this with a SQL
    query — the rest of the pipeline doesn't change.
    """
    trigger = cfg.get("trigger") or {}
    source = trigger.get("source")
    if not source:
        raise HTTPException(
            status_code=400,
            detail="config.yaml is missing trigger.source",
        )

    source_path = Path(source)
    if not source_path.is_absolute():
        # Resolve relative to the config file's directory so paths in
        # config.yaml don't depend on where the user invoked uvicorn from.
        source_path = (cfg["_paths"]["config_dir"] / source_path).resolve()

    if not source_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"trigger.source not found: {source_path}",
        )

    df = pd.read_csv(source_path)
    timestamp_col = cfg["input"]["timestamp_column"]
    mode = (trigger.get("mode") or "all").lower()

    if mode == "all":
        return df
    if mode == "time_range":
        start = trigger.get("start_time")
        end = trigger.get("end_time")
        if not start or not end:
            raise HTTPException(
                status_code=400,
                detail="trigger.mode=time_range requires start_time and end_time",
            )
        return slice_by_time(df, start, end, timestamp_column=timestamp_col)
    if mode == "last_n":
        n = trigger.get("last_n")
        if n is None:
            raise HTTPException(
                status_code=400,
                detail="trigger.mode=last_n requires last_n",
            )
        return slice_last_n_rows(df, int(n))

    raise HTTPException(status_code=400, detail=f"unknown trigger.mode: {mode}")


def _context_window_filename(slice_df: pd.DataFrame, timestamp_col: str) -> str:
    """Build the filename Winardi specified: context_window_<start>_<end>.csv."""
    if timestamp_col not in slice_df.columns or slice_df.empty:
        # Fallback when there's no timestamp to anchor on.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"context_window_{stamp}.csv"
    times = pd.to_datetime(slice_df[timestamp_col])
    fmt = "%Y%m%dT%H%M%S"
    return f"context_window_{times.min().strftime(fmt)}_{times.max().strftime(fmt)}.csv"


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
def root() -> dict:
    """Service metadata + endpoint index."""
    return {
        "service": "PCIL Job Orchestrator",
        "version": app.version,
        "endpoints": {
            "pipeline": ["POST /pipeline/run", "POST /pipeline/save_csv"],
            "anomaly": ["POST /anomaly/score"],
            "docs": "GET /docs",
        },
    }


@app.post("/pipeline/run", tags=["pipeline"])
def run_pipeline(req: RunPipelineRequest) -> dict:
    """Execute the full PCIL pipeline.

    Stages:
      1. Pull a shop-floor slice (trigger).
      2. Preprocess into the Golden DataFrame.
      3. Adapter -> X, y arrays.
      4. Fit the context model and produce the impacts dict.
      5. RAG retrieval + LLM composition (placeholders for now).

    Returns the impacts dict plus a placeholder LLM recommendation.
    """
    _, cfg = _resolve_config(req.config_path)

    slice_df = _pull_slice(cfg)
    if slice_df.empty:
        raise HTTPException(
            status_code=400,
            detail="trigger returned 0 rows; check trigger parameters",
        )

    golden_df, _fitted_preprocessor = preprocess(slice_df, cfg)

    targets, features = column_names_from_config(cfg, golden_df)
    _bundle = adapt(golden_df, targets, features)

    impacts, model = train_context_model_from_df(golden_df, cfg)

    artifact_paths: dict[str, str] = {}
    if req.persist:
        json_path, pkl_path = save_artifacts(
            impacts, model, cfg["_paths"]["output"],
            feature_names=features, target_names=targets,
        )
        artifact_paths = {"impacts_json": str(json_path), "model_pkl": str(pkl_path)}

    return {
        "status": "ok",
        "input_rows": int(len(slice_df)),
        "golden_rows": int(len(golden_df)),
        "impacts": impacts,
        "recovery_records": [],  # Pipeline #3 (RAG) — Robin
        "operator_recommendation": (
            "<LLM composer not wired yet — see deliverables/Week3/todo.md>"
        ),
        "artifacts": artifact_paths,
    }


@app.post("/pipeline/save_csv", tags=["pipeline"])
def save_csv(req: SaveCsvRequest) -> dict:
    """Pull the slice described in config.yaml and write it as
    `context_window_<start>_<end>.csv` under the config's output_dir.

    Separate from /pipeline/run so the normal runtime path doesn't
    incur disk IO. Use this when you want to inspect the raw slice.
    """
    _, cfg = _resolve_config(req.config_path)
    slice_df = _pull_slice(cfg)

    timestamp_col = cfg["input"]["timestamp_column"]
    filename = _context_window_filename(slice_df, timestamp_col)
    out_path = cfg["_paths"]["output"] / filename
    slice_df.to_csv(out_path, index=False)

    return {
        "status": "ok",
        "path": str(out_path),
        "rows": int(len(slice_df)),
    }


def _anomaly_bundle_path(model_type: str, model_id: str) -> Path:
    """Convention: bundles live at <PROJECT_ROOT>/data/<model_type>_<model_id>.pkl.

    Matches Zi Hin's `non_cyclical_config.yaml` output naming
    (`data/non_cyclical_inkjet_01.pkl`).
    """
    return PROJECT_ROOT / "data" / f"{model_type}_{model_id}.pkl"


def _score_non_cyclical(req: "AnomalyScoreRequest") -> dict:
    model_id = req.model_id or "inkjet_01"
    bundle_path = _anomaly_bundle_path("non_cyclical", model_id)
    if not bundle_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"non_cyclical bundle not found at {bundle_path}. "
                f"Train one first via `python pcil/utils/anomaly/non_cyclical/run.py`."
            ),
        )

    bundle = joblib.load(bundle_path)
    df = pd.DataFrame(req.data)

    # The score() function needs the channel columns the bundle was trained on.
    required_cols = set(bundle["channel_columns"])
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input rows missing required channel columns: {sorted(missing)}. "
                f"Bundle expects: {sorted(required_cols)}."
            ),
        )

    scored = non_cyclical_score(df, bundle)

    return {
        "status": "ok",
        "model_type": "non_cyclical",
        "model_id": model_id,
        "input_rows": int(len(df)),
        "windows_scored": int(len(scored)),
        "anomaly_scores": scored["anomaly_score"].tolist(),
        "window_starts": scored["window_start_idx"].tolist(),
        "bundle_path": str(bundle_path),
    }


def _score_cyclical(req: "AnomalyScoreRequest") -> dict:
    model_id = req.model_id or "inkjet_01"
    bundle_path = _anomaly_bundle_path("cyclical", model_id)
    if not bundle_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"cyclical bundle not found at {bundle_path}. "
                f"Train one via `python -m pcil.utils.anomaly.cyclical.prepare_data` "
                f"then `python -m pcil.utils.anomaly.cyclical.train`."
            ),
        )

    bundle = joblib.load(bundle_path)
    df = pd.DataFrame(req.data)

    # cyclical.score() needs signal_column + timestamp_column + machine_id_column
    # all present on the input rows.
    required_cols = {
        bundle["signal_column"],
        bundle["timestamp_column"],
        bundle["machine_id_column"],
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input rows missing required columns: {sorted(missing)}. "
                f"Bundle expects: {sorted(required_cols)}."
            ),
        )

    scored = cyclical_score(df, bundle)

    return {
        "status": "ok",
        "model_type": "cyclical",
        "model_id": model_id,
        "input_rows": int(len(df)),
        "cycles_scored": int(len(scored)),
        "anomaly_scores": scored["anomaly_score"].tolist(),
        "cycle_start_timestamps": (
            scored["cycle_start_timestamp"].astype(str).tolist()
            if "cycle_start_timestamp" in scored.columns
            else None
        ),
        "bundle_path": str(bundle_path),
    }


@app.post("/anomaly/score", tags=["anomaly"])
def anomaly_score(req: AnomalyScoreRequest) -> dict:
    """Engineer-facing API for anomaly scoring.

    Per Winardi (22 May): the anomaly modules are input -> output only.
    Engineers call this endpoint with raw time-series data and receive
    an anomaly score back. They then write the score into the correct
    row of their shop-floor DB themselves (only they know the mapping).

    Non-cyclical: wired to Zi Hin's RandomForestModel (recall ~0.68
    against the labelled acoustic dataset). Needs a trained bundle on
    disk at `data/non_cyclical_<model_id>.pkl`.

    Cyclical: wired to Jaymon's IsolationForestModel (peak slicing +
    waveform features). Needs a trained bundle on disk at
    `data/cyclical_<model_id>.pkl`.
    """
    if req.model_type == "non_cyclical":
        return _score_non_cyclical(req)
    if req.model_type == "cyclical":
        return _score_cyclical(req)
    # Should never reach here — Literal type bound catches it at request parse.
    raise HTTPException(status_code=400, detail=f"unknown model_type: {req.model_type}")

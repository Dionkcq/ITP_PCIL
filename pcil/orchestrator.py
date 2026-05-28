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
    POST /pipeline/run_csv    — same flow, but the slice arrives as an
                                 uploaded CSV instead of via config.yaml's
                                 trigger.source. For engineers who have
                                 a one-off CSV from the factory floor.
    POST /pipeline/save_csv   — pull a slice and write
                                 `context_window_<start>_<end>.csv`.
                                 Optional; not called during /run.

  /anomaly/*     Engineer-facing APIs.
    POST /anomaly/train       — train + persist a model bundle from
                                 uploaded CSVs.
    POST /anomaly/score       — input: time-series data + model_type.
                                 Output: anomaly score per window/cycle.

Run from PCIL_dev/:
    pip install -r requirements.txt
    uvicorn pcil.orchestrator:app --reload --host 0.0.0.0 --port 8000

OpenAPI / Swagger UI:  http://localhost:8000/docs
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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


async def _read_upload_to_df(upload: UploadFile) -> pd.DataFrame:
    """Read a FastAPI UploadFile into a pandas DataFrame.

    Raises HTTPException(400) if the file is empty, can't be parsed as
    CSV, or parses to an empty DataFrame. Keeps the error messages
    user-friendly for engineers calling the API.
    """
    contents = await upload.read()
    if not contents:
        raise HTTPException(
            status_code=400,
            detail=f"uploaded file '{upload.filename}' is empty",
        )
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:  # noqa: BLE001 — surface any pandas parse error
        raise HTTPException(
            status_code=400,
            detail=f"could not parse '{upload.filename}' as CSV: {exc}",
        ) from exc
    if df.empty:
        raise HTTPException(
            status_code=400,
            detail=f"uploaded CSV '{upload.filename}' contained no rows",
        )
    return df


def _run_pipeline_on_df(
    slice_df: pd.DataFrame,
    cfg: dict,
    *,
    persist: bool,
) -> dict:
    """Shared pipeline execution: preprocess -> adapter -> context model.

    Used by both /pipeline/run (data from config.trigger.source) and
    /pipeline/run_csv (data from upload). Keeping the body in one place
    means we can't drift between the two entry points.

    Schema mismatches (missing columns, NaN values, features out of
    range) get surfaced to the caller as HTTP 400 — they're client-side
    input problems, not server crashes.
    """
    if slice_df.empty:
        raise HTTPException(
            status_code=400,
            detail="input slice is empty; nothing to process",
        )

    try:
        golden_df, _fitted_preprocessor = preprocess(slice_df, cfg)
        targets, features = column_names_from_config(cfg, golden_df)
        _bundle = adapt(golden_df, targets, features)
        impacts, model = train_context_model_from_df(golden_df, cfg)
    except ValueError as exc:
        # preprocess/adapter raise ValueError on schema problems. Convert
        # to a clean 400 instead of leaking a 500 with a stack trace.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    artifact_paths: dict[str, str] = {}
    if persist:
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
            "pipeline": [
                "POST /pipeline/run",
                "POST /pipeline/run_csv",
                "POST /pipeline/save_csv",
            ],
            "anomaly": [
                "POST /anomaly/train",
                "POST /anomaly/score",
            ],
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
    return _run_pipeline_on_df(slice_df, cfg, persist=req.persist)


@app.post("/pipeline/run_csv", tags=["pipeline"])
async def run_pipeline_csv(
    file: UploadFile = File(
        ...,
        description="Shop-floor CSV slice. Same schema config.yaml expects.",
    ),
    config_path: str = Form(
        "machines/inkjet_printer/config.yaml",
        description="Config recipe path — relative to orchestrator's CWD.",
    ),
    persist: bool = Form(
        False,
        description=(
            "If true, also write context_model.pkl + impacts JSON to "
            "the output directory from config.yaml."
        ),
    ),
) -> dict:
    """Run the full pipeline against an uploaded CSV.

    Same downstream code as /pipeline/run; only difference is the slice
    arrives via multipart upload instead of being pulled from
    cfg['trigger']['source']. Useful when the engineer has a one-off
    CSV from the factory floor and doesn't want to edit config.yaml.

    The uploaded CSV must satisfy the schema declared in
    config.yaml's `input` block (timestamp_column, numerical_features,
    categorical_features, targets).
    """
    df = await _read_upload_to_df(file)
    _, cfg = _resolve_config(config_path)
    return _run_pipeline_on_df(df, cfg, persist=persist)


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
    if not req.data:
        raise HTTPException(
            status_code=400,
            detail="'data' must contain at least one row of sensor data",
        )

    model_id = req.model_id or "inkjet_01"
    bundle_path = _anomaly_bundle_path("non_cyclical", model_id)
    if not bundle_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"non_cyclical bundle not found at {bundle_path}. "
                "Train one first via POST /anomaly/train (model_type=non_cyclical, "
                "training_mode=clean_vs_anomaly) or the run.py CLI."
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
    if not req.data:
        raise HTTPException(
            status_code=400,
            detail="'data' must contain at least one row of sensor data",
        )

    model_id = req.model_id or "inkjet_01"
    bundle_path = _anomaly_bundle_path("cyclical", model_id)
    if not bundle_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"cyclical bundle not found at {bundle_path}. "
                "Train one first via POST /anomaly/train (model_type=cyclical, "
                "training_mode=normal_only) or the cyclical train.py CLI."
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


@app.post("/anomaly/train", tags=["anomaly"])
async def anomaly_train(
    model_type: Literal["cyclical", "non_cyclical"] = Form(
        ..., description="Which subpackage to train."),
    training_mode: Literal["normal_only", "clean_vs_anomaly"] = Form(
        ..., description=(
            "cyclical accepts 'normal_only'; non_cyclical accepts "
            "'clean_vs_anomaly'.")),
    model_id: str = Form(
        "inkjet_01",
        description="Identifier used in the saved bundle filename."),
    file: UploadFile | None = File(
        None,
        description="cyclical normal_only: training CSV with the signal column."),
    clean_file: UploadFile | None = File(
        None,
        description="non_cyclical clean_vs_anomaly: clean recording CSV."),
    anomaly_file: UploadFile | None = File(
        None,
        description="non_cyclical clean_vs_anomaly: anomaly recording CSV."),
    model_name: str = Form(
        "isolation_forest",
        description="Cyclical only. One of: z_score | isolation_forest | "
                    "one_class_svm | autoencoder. Most are stubs; "
                    "isolation_forest is the working default."),
    machine_id_column: str = Form(
        "machine_id",
        description="Cyclical only — column containing the machine identifier."),
    signal_column: str = Form(
        "signal_value",
        description="Cyclical only — column containing the cyclic signal."),
    timestamp_column: str = Form(
        "timestamp",
        description="Cyclical only — column containing the timestamps."),
    window_size_rows: int = Form(
        12800,
        description="Non-cyclical only — rows per fixed window. "
                    "Default 12800 = 0.5 s at 25.6 kHz."),
    train_ratio: float = Form(
        0.8,
        description="Non-cyclical only — fraction held for training."),
) -> dict:
    """Train an anomaly model from uploaded CSVs and persist the bundle.

    Two supported combinations:

    Combination A — Cyclical
        model_type=cyclical
        training_mode=normal_only
        file=<CSV with machine_id_column, signal_column, timestamp_column>

        Uses Jaymon's IsolationForest pipeline. Cycle detection runs on
        signal_column; per-cycle features feed an unsupervised model.

    Combination B — Non-cyclical
        model_type=non_cyclical
        training_mode=clean_vs_anomaly
        clean_file=<CSV with channel columns: Acceleration 0/1/2, AE>
        anomaly_file=<CSV with same channel columns>

        Uses Zi Hin's supervised RandomForest pipeline. Fixed windows
        on each recording; clean=0, anomaly=1; per-machine z-score
        normaliser fit on clean training only.

    The bundle is saved to:
        <PROJECT_ROOT>/data/<model_type>_<model_id>.pkl

    /anomaly/score then loads from that same path.
    """
    bundle_path = _anomaly_bundle_path(model_type, model_id)

    if model_type == "cyclical" and training_mode == "normal_only":
        if file is None:
            raise HTTPException(
                status_code=400,
                detail=("cyclical normal_only requires the 'file' form field "
                        "(the training CSV)."),
            )
        df = await _read_upload_to_df(file)
        required = {machine_id_column, signal_column, timestamp_column}
        missing = required - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"uploaded CSV is missing required columns: {sorted(missing)}. "
                    f"Expected columns: {sorted(required)}. "
                    "Adjust the form fields machine_id_column / signal_column / "
                    "timestamp_column if your headers differ."
                ),
            )
        from pcil.utils.anomaly.cyclical.train import train as cyclical_train_fn
        bundle = cyclical_train_fn(
            df,
            model_name=model_name,
            machine_id_column=machine_id_column,
            signal_column=signal_column,
            timestamp_column=timestamp_column,
        )
        input_rows = int(len(df))

    elif model_type == "non_cyclical" and training_mode == "clean_vs_anomaly":
        if clean_file is None or anomaly_file is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "non_cyclical clean_vs_anomaly requires both "
                    "'clean_file' and 'anomaly_file' form fields."
                ),
            )
        clean_df = await _read_upload_to_df(clean_file)
        anomaly_df = await _read_upload_to_df(anomaly_file)
        from pcil.utils.anomaly.non_cyclical.train import (
            train_from_clean_and_anomaly,
        )
        try:
            bundle = train_from_clean_and_anomaly(
                clean_df, anomaly_df,
                machine_id=model_id,
                window_size_rows=window_size_rows,
                train_ratio=train_ratio,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        input_rows = int(len(clean_df) + len(anomaly_df))

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported combination: model_type={model_type!r}, "
                f"training_mode={training_mode!r}. "
                "Supported: 'cyclical' + 'normal_only', or "
                "'non_cyclical' + 'clean_vs_anomaly'."
            ),
        )

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)

    return {
        "status": "ok",
        "model_type": model_type,
        "model_id": model_id,
        "training_mode": training_mode,
        "input_rows": input_rows,
        "bundle_path": str(bundle_path),
        "message": "Model trained and ready for /anomaly/score",
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

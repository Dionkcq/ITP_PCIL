"""Anomaly detection API (/anomaly/*) as a FastAPI APIRouter.

Isolated from the main orchestrator so the lightweight pipeline image never
imports it - and therefore never needs torch. It is mounted by:
  - orchestrator.py when PCIL_SERVICE=full (single container), and
  - anomaly_app.py (the standalone anomaly service, PCIL_SERVICE=anomaly).

Importing this module pulls the cyclical/non_cyclical/irregular score functions
(and torch, via the cyclical 1D-CNN autoencoder).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from pcil.runtime import PROJECT_ROOT, read_upload_to_df
from pcil.utils.anomaly.cyclical.score import score as cyclical_score
from pcil.utils.anomaly.irregular.score import score as irregular_score
from pcil.utils.anomaly.non_cyclical.score import score as non_cyclical_score

router = APIRouter()


class AnomalyScoreRequest(BaseModel):
    data: list[dict[str, Any]] = Field(
        ...,
        description="Time-series rows. Schema depends on the anomaly model.",
    )
    model_type: Literal["cyclical", "non_cyclical", "irregular"] = Field(
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


# Bundle filenames follow <model_type>_<model_id>.pkl; model_type itself
# contains an underscore for non_cyclical, so match the known prefixes.
_BUNDLE_STEM_RE = re.compile(r"^(non_cyclical|cyclical|irregular)_(.+)$")


@router.get("/anomaly/models", tags=["anomaly"])
def list_anomaly_models() -> dict:
    """List the trained anomaly bundles available in data/.

    Drives the dashboard's bundle indicator: a model_type + model_id
    combination is scoreable only when its .pkl appears here. Filename
    metadata only — bundles are not unpickled (torch bundles are heavy).
    """
    models = []
    data_dir = PROJECT_ROOT / "data"
    if data_dir.is_dir():
        for p in sorted(data_dir.glob("*.pkl")):
            m = _BUNDLE_STEM_RE.match(p.stem)
            if not m:
                continue
            stat = p.stat()
            models.append({
                "model_type": m.group(1),
                "model_id": m.group(2),
                "file": p.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(timespec="seconds"),
            })
    return {"models": models}


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

    try:
        scored = non_cyclical_score(df, bundle)
    except (KeyError, ValueError) as exc:
        # e.g. an unknown machine_id at score time (PerMachineNormaliser) or a
        # shape/column problem — a client input error, not a server crash.
        raise HTTPException(
            status_code=400, detail=f"non_cyclical scoring failed: {exc}"
        ) from exc

    return {
        "status": "ok",
        "model_type": "non_cyclical",
        "model_id": model_id,
        "input_rows": int(len(df)),
        "windows_scored": int(len(scored)),
        "anomaly_scores": scored["anomaly_score"].tolist(),
        "is_anomaly": None,
        "threshold": None,
        "threshold_source": "not_configured",
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

    try:
        scored = cyclical_score(df, bundle)
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail=f"cyclical scoring failed: {exc}"
        ) from exc

    threshold = bundle.get("threshold")
    threshold_source = (
        "bundle_95th_percentile"
        if threshold is not None
        else "score_median_fallback"
    )
    if threshold is None and "anomaly_score" in scored.columns and not scored.empty:
        threshold = float(scored["anomaly_score"].median())

    return {
        "status": "ok",
        "model_type": "cyclical",
        "model_id": model_id,
        "input_rows": int(len(df)),
        "cycles_scored": int(len(scored)),
        "anomaly_scores": scored["anomaly_score"].tolist(),
        "is_anomaly": (
            scored["is_anomaly"].tolist()
            if "is_anomaly" in scored.columns
            else None
        ),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "cycle_start_timestamps": (
            scored["cycle_start_timestamp"].astype(str).tolist()
            if "cycle_start_timestamp" in scored.columns
            else None
        ),
        "bundle_path": str(bundle_path),
    }


def _score_irregular(req: "AnomalyScoreRequest") -> dict:
    if not req.data:
        raise HTTPException(
            status_code=400,
            detail="'data' must contain at least one row of sensor data",
        )

    model_id = req.model_id or "inkjet_01"
    bundle_path = _anomaly_bundle_path("irregular", model_id)
    if not bundle_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"irregular bundle not found at {bundle_path}. "
                "Train one first via POST /anomaly/train (model_type=irregular, "
                "training_mode=normal_only) or the irregular train.py CLI."
            ),
        )

    bundle = joblib.load(bundle_path)
    df = pd.DataFrame(req.data)

    # irregular.score() needs timestamp_column + machine_id_column
    # (+ value_column when the bundle was trained with one).
    required_cols = {
        bundle["timestamp_column"],
        bundle["machine_id_column"],
    }
    if bundle.get("value_column"):
        required_cols.add(bundle["value_column"])
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input rows missing required columns: {sorted(missing)}. "
                f"Bundle expects: {sorted(required_cols)}."
            ),
        )

    try:
        scored = irregular_score(df, bundle)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    threshold = bundle.get("threshold")
    threshold_source = (
        "bundle_95th_percentile"
        if threshold is not None
        else "score_median_fallback"
    )
    if threshold is None and "anomaly_score" in scored.columns and not scored.empty:
        threshold = float(scored["anomaly_score"].median())

    return {
        "status": "ok",
        "model_type": "irregular",
        "model_id": model_id,
        "input_rows": int(len(df)),
        "windows_scored": int(len(scored)),
        "anomaly_scores": scored["anomaly_score"].tolist(),
        "is_anomaly": scored["is_anomaly"].tolist(),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "window_start_timestamps": (
            scored["window_start_timestamp"].astype(str).tolist()
        ),
        "bundle_path": str(bundle_path),
    }


@router.post("/anomaly/train", tags=["anomaly"])
async def anomaly_train(
    model_type: Literal["cyclical", "non_cyclical", "irregular"] = Form(
        ..., description="Which subpackage to train."),
    training_mode: Literal["normal_only", "clean_vs_anomaly"] = Form(
        ..., description=(
            "cyclical and irregular accept 'normal_only'; non_cyclical "
            "accepts 'clean_vs_anomaly'.")),
    model_id: str = Form(
        "inkjet_01",
        description="Identifier used in the saved bundle filename."),
    file: UploadFile | None = File(
        None,
        description="cyclical/irregular normal_only: training CSV."),
    clean_file: UploadFile | None = File(
        None,
        description="non_cyclical clean_vs_anomaly: clean recording CSV."),
    anomaly_file: UploadFile | None = File(
        None,
        description="non_cyclical clean_vs_anomaly: anomaly recording CSV."),
    model_name: str = Form(
        "autoencoder",
        description="Cyclical only. One of: isolation_forest | autoencoder. "
                    "autoencoder (1D CNN) is the selected default; "
                    "isolation_forest is the lightweight fallback."),
    machine_id_column: str = Form(
        "machine_id",
        description="Cyclical/irregular — column containing the machine identifier."),
    signal_column: str = Form(
        "signal_value",
        description="Cyclical only — column containing the cyclic signal."),
    timestamp_column: str = Form(
        "timestamp",
        description="Cyclical/irregular — column containing the timestamps."),
    value_column: str | None = Form(
        None,
        description="Irregular only — optional numeric column for value_* "
                    "features. Omit for pure event logs."),
    window_seconds: float = Form(
        1.0,
        description="Irregular only — window duration in wall-clock seconds."),
    window_size_rows: int = Form(
        12800,
        description="Non-cyclical only — rows per fixed window. "
                    "Default 12800 = 0.5 s at 25.6 kHz."),
    train_ratio: float = Form(
        0.8,
        description="Non-cyclical only — fraction held for training."),
    header_skiprows: int = Form(
        0,
        description=(
            "Rows to skip before the CSV header. Set to 5 for raw WebDAQ "
            "acoustic exports (non_cyclical), whose file starts with a "
            "device-info preamble before the 'Sample,Time (s),Acceleration ...' "
            "header. Leave 0 for already-clean CSVs.")),
) -> dict:
    """Train an anomaly model from uploaded CSVs and persist the bundle.

    Two supported combinations:

    Combination A — Cyclical
        model_type=cyclical
        training_mode=normal_only
        file=<CSV with machine_id_column, signal_column, timestamp_column>

        Uses Jaymon's 1D CNN autoencoder by default (model_name=autoencoder),
        with isolation_forest as a lightweight fallback. Cycle detection runs
        on signal_column; per-cycle waveforms feed an unsupervised model.

    Combination B — Non-cyclical
        model_type=non_cyclical
        training_mode=clean_vs_anomaly
        clean_file=<CSV with channel columns: Acceleration 0/1/2, AE>
        anomaly_file=<CSV with same channel columns>

        Uses Zi Hin's supervised RandomForest pipeline. Fixed windows
        on each recording; clean=0, anomaly=1; per-machine z-score
        normaliser fit on clean training only.

    Combination C — Irregular
        model_type=irregular
        training_mode=normal_only
        file=<CSV with machine_id_column, timestamp_column
              (+ optional value_column)>

        For irregularly-sampled data (event logs, on-change sensors).
        Fixed-duration time windows; event-rate + inter-arrival-gap
        features; unsupervised IsolationForest.

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
        df = await read_upload_to_df(file, skiprows=header_skiprows)
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
        clean_df = await read_upload_to_df(clean_file, skiprows=header_skiprows)
        anomaly_df = await read_upload_to_df(anomaly_file, skiprows=header_skiprows)
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

    elif model_type == "irregular" and training_mode == "normal_only":
        if file is None:
            raise HTTPException(
                status_code=400,
                detail=("irregular normal_only requires the 'file' form field "
                        "(the training CSV)."),
            )
        df = await read_upload_to_df(file, skiprows=header_skiprows)
        required = {machine_id_column, timestamp_column}
        if value_column:
            required.add(value_column)
        missing = required - set(df.columns)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"uploaded CSV is missing required columns: {sorted(missing)}. "
                    f"Expected columns: {sorted(required)}. "
                    "Adjust the form fields machine_id_column / timestamp_column / "
                    "value_column if your headers differ."
                ),
            )
        from pcil.utils.anomaly.irregular.train import train as irregular_train_fn
        try:
            bundle = irregular_train_fn(
                df,
                machine_id_column=machine_id_column,
                timestamp_column=timestamp_column,
                value_column=value_column or None,
                window_seconds=window_seconds,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        input_rows = int(len(df))

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported combination: model_type={model_type!r}, "
                f"training_mode={training_mode!r}. "
                "Supported: 'cyclical' + 'normal_only', "
                "'non_cyclical' + 'clean_vs_anomaly', or "
                "'irregular' + 'normal_only'."
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


@router.post("/anomaly/score", tags=["anomaly"])
def anomaly_score(req: AnomalyScoreRequest) -> dict:
    """Engineer-facing API for anomaly scoring.

    Per Winardi (22 May): the anomaly modules are input -> output only.
    Engineers call this endpoint with raw time-series data and receive
    an anomaly score back. They then write the score into the correct
    row of their shop-floor DB themselves (only they know the mapping).

    Non-cyclical: wired to Zi Hin's RandomForestModel (recall ~0.68
    against the labelled acoustic dataset). Needs a trained bundle on
    disk at `data/non_cyclical_<model_id>.pkl`.

    Cyclical: wired to Jaymon's 1D CNN autoencoder (peak slicing +
    waveform features). Needs a trained bundle on disk at
    `data/cyclical_<model_id>.pkl`.

    Irregular: fixed-duration time windows + arrival-pattern features +
    IsolationForest, for irregularly-sampled data (event logs,
    on-change sensors). Needs a trained bundle on disk at
    `data/irregular_<model_id>.pkl`.
    """
    if req.model_type == "non_cyclical":
        return _score_non_cyclical(req)
    if req.model_type == "cyclical":
        return _score_cyclical(req)
    if req.model_type == "irregular":
        return _score_irregular(req)
    # Should never reach here — Literal type bound catches it at request parse.
    raise HTTPException(status_code=400, detail=f"unknown model_type: {req.model_type}")

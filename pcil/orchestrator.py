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
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pcil.adapter import adapt, column_names_from_config
from pcil.preprocess import load_config, preprocess
from pcil.train_context_model import save_artifacts, train_context_model_from_df
from pcil.trigger import slice_by_time, slice_last_n_rows
# RAG pipeline - imported at module level; guarded so missing
# google-genai does not break non-RAG endpoints or the test suite.
from pcil.rag.composer import compose_recommendation
from pcil.rag.loader import load_all_recovery_docs
from pcil.rag.lookup import lookup_keywords
from pcil.utils.anomaly.cyclical.score import score as cyclical_score
from pcil.utils.anomaly.irregular.score import score as irregular_score
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
RAG_DIR = PROJECT_ROOT / "data" / "RAG"

# Built dashboard assets. In Docker, PCIL_PROJECT_ROOT=/app and the
# multi-stage build copies the bundle to /app/dashboard/dist. In local
# development, PROJECT_ROOT intentionally points one level above PCIL/ so
# data/ resolves correctly; REPO_ROOT points at the folder containing this
# code checkout and catches PCIL/dashboard/dist after `npm run build`.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DASHBOARD_DIST_CANDIDATES = (
    PROJECT_ROOT / "dashboard" / "dist",
    REPO_ROOT / "dashboard" / "dist",
)
DASHBOARD_URL_PATH = "/dashboard"

app = FastAPI(
    title="PCIL Job Orchestrator",
    version="0.1.0",
    description=(
        "Coordinates the PCIL runtime pipeline. See /docs for the "
        "interactive Swagger UI."
    ),
)

# Allow the React dashboard (a separate origin/port) to call the API from a
# browser. Permissive for the PoC; tighten allow_origins for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _rag_backend() -> str:
    return os.environ.get("RAG_BACKEND", "file").lower()


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@app.on_event("startup")
def warm_postgres_rag() -> None:
    """Warm PostgreSQL RAG caches when the dockerized backend is enabled."""
    if _rag_backend() != "postgres":
        return
    strict = _bool_env("RAG_STRICT_STARTUP", False)
    try:
        from pcil.rag.db import run_migrations
        from pcil.rag.embeddings import EmbeddingModelCache
        from pcil.rag.hybrid import BM25IndexCache
        from pcil.rag.ingest import ingest_rag_dir

        run_migrations()
        if _bool_env("RAG_AUTO_INGEST", True) and RAG_DIR.is_dir():
            ingest_rag_dir(RAG_DIR)
        else:
            BM25IndexCache.rebuild()
        if _bool_env("RAG_WARM_EMBEDDINGS", True):
            EmbeddingModelCache.warm_up()
    except Exception:
        if strict:
            raise


def attach_dashboard(app: FastAPI, dist_dir: Path | str | None = None) -> Path | None:
    """Serve the built dashboard at /dashboard if static assets are present.

    Resolution order:
      1. explicit `dist_dir` argument
      2. DASHBOARD_DIST_DIR environment variable
      3. <PROJECT_ROOT>/dashboard/dist (Docker default)
      4. <repo checkout>/dashboard/dist (local dev fallback)

    Returns the resolved path when the mount succeeded, otherwise None
    (so the orchestrator still boots when running the API without a
    built dashboard, e.g. in CI or before `npm run build`).
    """
    if dist_dir or os.environ.get("DASHBOARD_DIST_DIR"):
        candidates = (Path(dist_dir or os.environ["DASHBOARD_DIST_DIR"]),)
    else:
        candidates = DEFAULT_DASHBOARD_DIST_CANDIDATES

    for candidate in candidates:
        dist_path = Path(candidate)
        if dist_path.is_dir() and (dist_path / "index.html").is_file():
            app.mount(
                DASHBOARD_URL_PATH,
                StaticFiles(directory=dist_path, html=True),
                name="dashboard",
            )
            return dist_path
    return None


DASHBOARD_DIST = attach_dashboard(app)


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


class TrainBaselineRequest(BaseModel):
    config_path: str = Field(..., examples=["machines/inkjet_printer/config.yaml"])


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
    tried: list[Path] = [source_path]
    if not source_path.is_absolute():
        # Relative paths resolve against PROJECT_ROOT first — the same
        # rule as anomaly bundles and RAG_DIR, so "data/x.csv" works
        # identically in dev (PROJECT_ROOT = ITP/) and in Docker
        # (PCIL_PROJECT_ROOT=/app, data mounted at /app/data). Fall back
        # to the config file's directory for recipes written with
        # config-relative paths like "../../../data/x.csv".
        tried = [
            (PROJECT_ROOT / source_path).resolve(),
            (cfg["_paths"]["config_dir"] / source_path).resolve(),
        ]
        source_path = next((p for p in tried if p.is_file()), tried[0])

    if not source_path.is_file():
        also = [str(p) for p in tried if p != source_path]
        raise HTTPException(
            status_code=404,
            detail=(
                f"trigger.source not found: {source_path}"
                + (f" (also tried: {', '.join(also)})" if also else "")
            ),
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


def _baseline_artifact_paths(cfg: dict) -> dict[str, Path]:
    output = cfg["_paths"]["output"]
    return {
        "preprocessor": output / "baseline_preprocessor.pkl",
        "model": output / "baseline_context_model.pkl",
        "stats": output / "baseline_stats.json",
        "impacts": output / "baseline_context_model_impacts.json",
    }


def _load_baseline_preprocessor(cfg: dict) -> tuple[Any | None, list[str]]:
    paths = _baseline_artifact_paths(cfg)
    if not paths["preprocessor"].is_file():
        return None, ["baseline_preprocessor_missing"]
    try:
        return joblib.load(paths["preprocessor"]), []
    except Exception:  # noqa: BLE001
        return None, ["baseline_preprocessor_unreadable"]


def _load_baseline_stats(cfg: dict) -> tuple[dict | None, list[str]]:
    paths = _baseline_artifact_paths(cfg)
    if not paths["stats"].is_file():
        return None, ["baseline_stats_missing"]
    try:
        return json.loads(paths["stats"].read_text(encoding="utf-8")), []
    except Exception:  # noqa: BLE001
        return None, ["baseline_stats_unreadable"]


def _series_stats(df: pd.DataFrame, col: str) -> dict[str, Any]:
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
            "active_count": 0,
            "active_ratio": None,
        }
    active_count = int((values > 0).sum())
    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std(ddof=0)),
        "active_count": active_count,
        "active_ratio": float(active_count / values.count()),
    }


def _target_status(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value >= 0.85:
        return "good"
    if value >= 0.6:
        return "watch"
    return "degraded"


def _signal_summary(slice_df: pd.DataFrame, cfg: dict) -> dict[str, Any]:
    schema = cfg["input"]
    numerical = list(schema.get("numerical_features", []) or [])
    targets = list(schema.get("targets", []) or [])

    features = {
        col: _series_stats(slice_df, col)
        for col in numerical
        if col in slice_df.columns
    }
    target_stats: dict[str, Any] = {}
    for col in targets:
        if col not in slice_df.columns:
            continue
        stats = _series_stats(slice_df, col)
        target_stats[col] = {
            **stats,
            "status": _target_status(stats["mean"]),
        }
    return {
        "source": "raw_window",
        "features": features,
        "targets": target_stats,
    }


def _baseline_stats_from_df(slice_df: pd.DataFrame, cfg: dict) -> dict[str, Any]:
    timestamp_col = cfg["input"]["timestamp_column"]
    timestamps = pd.to_datetime(slice_df[timestamp_col])
    summary = _signal_summary(slice_df, cfg)
    return {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(slice_df)),
        "context_window": {
            "start_time": timestamps.min().isoformat(),
            "end_time": timestamps.max().isoformat(),
        },
        "features": summary["features"],
        "targets": summary["targets"],
    }


def _compare_to_baseline(
    signal_summary: dict[str, Any],
    baseline_stats: dict | None,
) -> dict[str, Any]:
    if not baseline_stats:
        return {
            "status": "not_available",
            "features": {},
            "targets": {},
        }

    def compare_group(group: str) -> dict[str, Any]:
        current_group = signal_summary.get(group, {})
        baseline_group = baseline_stats.get(group, {})
        compared: dict[str, Any] = {}
        for name, current in current_group.items():
            base = baseline_group.get(name)
            if not base:
                continue
            current_mean = current.get("mean")
            baseline_mean = base.get("mean")
            baseline_std = base.get("std")
            if current_mean is None or baseline_mean is None:
                continue
            deviation = float(current_mean - baseline_mean)
            z_score = (
                float(deviation / baseline_std)
                if baseline_std not in (None, 0)
                else None
            )
            direction = "within_baseline"
            if z_score is not None and z_score >= 1.0:
                direction = "above_baseline"
            elif z_score is not None and z_score <= -1.0:
                direction = "below_baseline"
            compared[name] = {
                "current_mean": float(current_mean),
                "baseline_mean": float(baseline_mean),
                "baseline_std": baseline_std,
                "deviation_from_baseline": deviation,
                "z_score": z_score,
                "direction": direction,
            }
        return compared

    return {
        "status": "available",
        "trained_at": baseline_stats.get("trained_at"),
        "features": compare_group("features"),
        "targets": compare_group("targets"),
    }


def _save_baseline_artifacts(
    cfg: dict,
    preprocessor: Any,
    model: Any,
    impacts: dict,
    baseline_stats: dict,
    *,
    feature_names: list[str],
    target_names: list[str],
) -> dict[str, str]:
    paths = _baseline_artifact_paths(cfg)
    paths["stats"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, paths["preprocessor"])
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "target_names": target_names,
            "model_role": "baseline_context_model",
        },
        paths["model"],
    )
    paths["impacts"].write_text(json.dumps(impacts, indent=2), encoding="utf-8")
    paths["stats"].write_text(
        json.dumps(baseline_stats, indent=2), encoding="utf-8",
    )
    return {name: str(path) for name, path in paths.items()}


def _retrieve_recovery_records(rag_query: str, *, top_k: int = 3) -> tuple[list[dict], int | None]:
    if _rag_backend() == "postgres":
        from pcil.rag.hybrid import hybrid_lookup, insert_search_event

        recovery_records, meta = hybrid_lookup(rag_query, top_k=top_k)
        search_event_id = insert_search_event(rag_query, meta)
        return recovery_records, search_event_id

    if not RAG_DIR.is_dir():
        raise FileNotFoundError(
            f"RAG document directory not found. Expected: {RAG_DIR}"
        )
    all_records = load_all_recovery_docs(RAG_DIR)
    return lookup_keywords(rag_query, all_records, top_k=top_k), None


def _audit_rag_recommendation(
    *,
    search_event_id: int | None,
    impacts: dict,
    signal_summary: dict,
    baseline_comparison: dict,
    operator_recommendation: str,
    recommendation_source: str,
    recommendation_warnings: list[str],
) -> int | None:
    if _rag_backend() != "postgres":
        return None
    try:
        from pcil.rag.hybrid import insert_recommendation_event

        return insert_recommendation_event(
            search_event_id=search_event_id,
            impacts=impacts,
            signal_summary=signal_summary,
            baseline_comparison=baseline_comparison,
            recommendation_text=operator_recommendation,
            recommendation_source=recommendation_source,
            recommendation_warnings=recommendation_warnings,
        )
    except Exception:  # noqa: BLE001
        return None


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

    response_warnings: list[str] = []
    baseline_preprocessor, baseline_preprocessor_warnings = _load_baseline_preprocessor(cfg)
    response_warnings.extend(baseline_preprocessor_warnings)
    signal_summary = _signal_summary(slice_df, cfg)
    baseline_stats, baseline_stats_warnings = _load_baseline_stats(cfg)
    response_warnings.extend(baseline_stats_warnings)
    baseline_comparison = _compare_to_baseline(signal_summary, baseline_stats)

    try:
        golden_df, _fitted_preprocessor = preprocess(
            slice_df,
            cfg,
            fitted_pipeline=baseline_preprocessor,
        )
        targets, features = column_names_from_config(cfg, golden_df)
        _bundle = adapt(golden_df, targets, features)
        impacts, model = train_context_model_from_df(golden_df, cfg)
        impacts["model_role"] = "window_correlation_model"
        impacts["preprocessing_source"] = (
            "baseline_preprocessor"
            if baseline_preprocessor is not None
            else "fit_on_current_window"
        )
        if baseline_preprocessor is None:
            response_warnings.append("features_scaled_on_current_window")
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

    # --- RAG retrieval + LLM composition ---------------------------
    # Retrieval and composition are guarded separately: retrieval is
    # fully local (DOCX files on disk), composition needs the Gemini
    # API. If only composition fails (no key, no internet, timeout),
    # the operator still gets the retrieved recovery records.
    rag_search_event_id: int | None = None
    rag_recommendation_id: int | None = None
    if _rag_backend() == "file" and not RAG_DIR.is_dir():
        recovery_records = []
        operator_recommendation = (
            "RAG document directory not found. "
            f"Expected: {RAG_DIR}. "
            "Mount the data/RAG/ folder and restart the orchestrator."
        )
        recommendation_source = "fallback"
        recommendation_warnings = ["rag_directory_missing"]
    else:
        try:
            rag_query = _build_rag_query(
                impacts,
                signal_summary=signal_summary,
                baseline_comparison=baseline_comparison,
            )
            recovery_records, rag_search_event_id = _retrieve_recovery_records(
                rag_query, top_k=3,
            )
        except Exception as exc:  # noqa: BLE001
            recovery_records = []
            operator_recommendation = (
                f"RAG retrieval failed ({type(exc).__name__}): {exc}. "
                "Review the impacts data manually."
            )
            recommendation_source = "retrieval_error"
            recommendation_warnings = ["rag_retrieval_failed"]
        else:
            if not recovery_records:
                operator_recommendation = compose_recommendation(
                    impacts,
                    recovery_records,
                    signal_summary=signal_summary,
                    baseline_comparison=baseline_comparison,
                )
                recommendation_source = "no_records"
                recommendation_warnings = ["no_matching_recovery_records"]
            else:
                try:
                    operator_recommendation = compose_recommendation(
                        impacts,
                        recovery_records,
                        signal_summary=signal_summary,
                        baseline_comparison=baseline_comparison,
                    )
                except Exception as exc:  # noqa: BLE001
                    operator_recommendation = (
                        f"LLM composition failed ({type(exc).__name__}): {exc}. "
                        "Review the recovery records below manually."
                    )
                    recommendation_source = "fallback"
                    recommendation_warnings = ["llm_composition_failed"]
                else:
                    if operator_recommendation.startswith("LLM composition failed"):
                        recommendation_source = "fallback"
                        recommendation_warnings = ["llm_composition_failed"]
                    elif operator_recommendation.startswith("LLM returned an empty response"):
                        recommendation_source = "fallback"
                        recommendation_warnings = ["llm_empty_response"]
                    else:
                        recommendation_source = "gemini"
                        recommendation_warnings = []
        rag_recommendation_id = _audit_rag_recommendation(
            search_event_id=rag_search_event_id,
            impacts=impacts,
            signal_summary=signal_summary,
            baseline_comparison=baseline_comparison,
            operator_recommendation=operator_recommendation,
            recommendation_source=recommendation_source,
            recommendation_warnings=recommendation_warnings,
        )
    # ---------------------------------------------------------------

    # Window-level mean of each target, for the dashboard KPI cards.
    target_summary = {
        t: float(golden_df[t].mean())
        for t in targets
        if t in golden_df.columns
    }

    return {
        "status": "ok",
        "input_rows": int(len(slice_df)),
        "golden_rows": int(len(golden_df)),
        "impacts": impacts,
        "target_summary": target_summary,
        "signal_summary": signal_summary,
        "baseline_comparison": baseline_comparison,
        "recovery_records": recovery_records,
        "operator_recommendation": operator_recommendation,
        "recommendation_source": recommendation_source,
        "recommendation_warnings": recommendation_warnings,
        "rag_backend": _rag_backend(),
        "rag_search_event_id": rag_search_event_id,
        "rag_recommendation_id": rag_recommendation_id,
        "pipeline_warnings": response_warnings,
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


def _feature_terms(feature: str) -> list[str]:
    return [t for t in feature.lower().split("_") if len(t) > 2]


def _direction_for_feature(
    feature: str,
    signal_summary: dict[str, Any] | None,
    baseline_comparison: dict[str, Any] | None,
) -> str | None:
    baseline = (baseline_comparison or {}).get("features", {}).get(feature, {})
    direction = baseline.get("direction")
    if direction == "above_baseline":
        return "high"
    if direction == "below_baseline":
        return "low"

    stats = (signal_summary or {}).get("features", {}).get(feature, {})
    mean = stats.get("mean")
    if mean is None:
        return None
    if feature.endswith("_low_ratio") and mean > 0:
        return "low"
    if feature.endswith("_present") and stats.get("active_count", 0) > 0:
        return "present"
    if "anomaly_score" in feature and mean > 0:
        return "anomaly"
    return None


def _build_rag_query(
    impacts: dict,
    *,
    signal_summary: dict[str, Any] | None = None,
    baseline_comparison: dict[str, Any] | None = None,
) -> str:
    """Build a signal-aware keyword query for RAG retrieval.

    Prefer live feature names and direction ("air pressure low",
    "vibration high", "OEE degraded") over static config prose. This
    keeps retrieval tied to what the current window is actually showing.
    """
    phrases: list[str] = []
    target_stats = (signal_summary or {}).get("targets", {})
    context_blocks = impacts.get("context", [])

    def target_key(block: dict) -> float:
        mean = target_stats.get(block["target"], {}).get("mean")
        return mean if mean is not None else block.get("intercept", 1.0)

    for block in sorted(context_blocks, key=target_key)[:2]:
        target = block["target"]
        status = target_stats.get(target, {}).get("status")
        if status in {"degraded", "watch"}:
            phrases.append(f"{target} degraded")
        else:
            phrases.append(target)

        ranked = block.get("ranked_feature_impacts", [])
        negative = [fi for fi in ranked if fi.get("raw_impact_score", 0) < 0]
        selected = (negative or ranked)[:3]
        for fi in selected:
            feature = fi["feature"]
            terms = _feature_terms(feature)
            direction = _direction_for_feature(
                feature,
                signal_summary,
                baseline_comparison,
            )
            if direction:
                phrases.append(" ".join([*terms, direction]))
            else:
                phrases.append(" ".join(terms))

    return " ".join(p for p in phrases if p)


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
                "POST /pipeline/train_baseline",
                "POST /pipeline/save_csv",
            ],
            "anomaly": [
                "POST /anomaly/train",
                "POST /anomaly/score",
                "GET /anomaly/models",
            ],
            "rag": [
                "POST /rag/reindex",
            ],
            "config": [
                "GET /configs",
                "GET /configs/load",
                "POST /configs/validate",
                "POST /configs/save",
                "POST /configs/create",
                "POST /configs/delete",
            ],
            "docs": "GET /docs",
            "dashboard": f"GET {DASHBOARD_URL_PATH}/",
        },
        "dashboard_available": DASHBOARD_DIST is not None,
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


@app.post("/pipeline/train_baseline", tags=["pipeline"])
def train_baseline(req: TrainBaselineRequest) -> dict:
    """Train and persist the normal-operation baseline artifacts.

    The configured trigger source should point at historical normal data.
    Later /pipeline/run calls reuse the saved preprocessor and compare
    live windows against the saved raw feature/target statistics.
    """
    _, cfg = _resolve_config(req.config_path)
    baseline_df = _pull_slice(cfg)
    if baseline_df.empty:
        raise HTTPException(
            status_code=400,
            detail="baseline slice is empty; nothing to train",
        )

    try:
        golden_df, fitted_preprocessor = preprocess(baseline_df, cfg)
        targets, features = column_names_from_config(cfg, golden_df)
        _bundle = adapt(golden_df, targets, features)
        impacts, model = train_context_model_from_df(golden_df, cfg)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    impacts["model_role"] = "baseline_context_model"
    impacts["preprocessing_source"] = "fit_on_baseline"
    baseline_stats = _baseline_stats_from_df(baseline_df, cfg)
    artifact_paths = _save_baseline_artifacts(
        cfg,
        fitted_preprocessor,
        model,
        impacts,
        baseline_stats,
        feature_names=features,
        target_names=targets,
    )

    return {
        "status": "ok",
        "baseline_rows": int(len(baseline_df)),
        "golden_rows": int(len(golden_df)),
        "impacts": impacts,
        "baseline_stats": baseline_stats,
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


# ─────────────────────────────────────────────────────────────
# Config recipe endpoints (dashboard config editor)
#
# The dashboard's Config tab edits recipes through these endpoints —
# never raw YAML. The server parses, validates, and re-serialises with
# yaml.safe_dump, so a bad submission is rejected with a list of errors
# instead of ever producing a corrupt file. Every overwrite stores a
# timestamped backup under machines/<machine>/.backups/.
# ─────────────────────────────────────────────────────────────

@app.post("/rag/reindex", tags=["rag"])
def reindex_rag() -> dict:
    """Rebuild the PostgreSQL RAG store from the mounted DOCX directory."""
    if _rag_backend() != "postgres":
        return {
            "status": "skipped",
            "rag_backend": _rag_backend(),
            "message": "RAG_BACKEND is not 'postgres'; file-based RAG is active.",
        }
    if not RAG_DIR.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"RAG document directory not found at {RAG_DIR}",
        )
    try:
        from pcil.rag.ingest import ingest_rag_dir

        result = ingest_rag_dir(RAG_DIR)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"RAG reindex failed ({type(exc).__name__}): {exc}",
        ) from exc
    return {"rag_backend": "postgres", **result}


# Root folder of per-machine recipes. CWD-relative to match how
# config_path strings resolve everywhere else (dev: PCIL_dev/machines,
# Docker: /app/machines). Mount the folder from the host in deployment
# (docker-compose does) so dashboard edits survive container recreation.
MACHINES_ROOT = (Path.cwd() / "machines").resolve()

_RECIPE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]{0,63}")

# Top-level sections the editor owns. Any other top-level key an
# engineer added by hand is preserved untouched on save.
_EDITABLE_SECTIONS = (
    "system", "pipeline", "trigger", "input", "feature_descriptions",
)


class SaveConfigRequest(BaseModel):
    path: str = Field(
        ...,
        description="Recipe to edit, relative to machines/ — e.g. 'inkjet_printer/config.yaml'.",
        examples=["inkjet_printer/config.yaml"],
    )
    config: dict = Field(
        ...,
        description="The full edited recipe (system / pipeline / trigger / input / feature_descriptions).",
    )
    save_as: str | None = Field(
        None,
        description=(
            "Optional new recipe name (letters, digits, '_', '-'). Saves "
            "alongside the original instead of overwriting it."
        ),
    )


def _safe_recipe_path(recipe: str, *, must_exist: bool = True) -> Path:
    """Resolve a recipe reference and confine it to MACHINES_ROOT.

    Accepts 'inkjet_printer/config.yaml' or the run-endpoint style
    'machines/inkjet_printer/config.yaml'. Rejects absolute paths,
    traversal ('..') and non-YAML suffixes so the editor endpoints can
    never read or write outside the machines folder.
    """
    rel = Path(recipe)
    if rel.is_absolute():
        raise HTTPException(
            status_code=400,
            detail="config path must be relative to the machines/ folder",
        )
    if rel.parts and rel.parts[0] == "machines":
        rel = Path(*rel.parts[1:])
    if rel.suffix.lower() not in (".yaml", ".yml"):
        raise HTTPException(status_code=400, detail="config path must end in .yaml")

    p = (MACHINES_ROOT / rel).resolve()
    try:
        p.relative_to(MACHINES_ROOT)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="config path must stay inside the machines/ folder",
        ) from None
    if must_exist and not p.is_file():
        raise HTTPException(status_code=404, detail=f"config not found: {recipe}")
    return p


def _normalize_recipe(cfg: Any) -> Any:
    """Strip whitespace and coerce obvious types before validation.

    Form submissions arrive as strings ('500' for last_n, padded column
    names); normalising here keeps the validator's error messages about
    real problems instead of formatting noise.
    """
    if not isinstance(cfg, dict):
        return cfg
    out = dict(cfg)

    if isinstance(out.get("system"), str):
        out["system"] = out["system"].strip()

    pipeline = out.get("pipeline")
    if isinstance(pipeline, dict):
        pipeline = dict(pipeline)
        if isinstance(pipeline.get("output_dir"), str):
            pipeline["output_dir"] = pipeline["output_dir"].strip()
        out["pipeline"] = pipeline

    trigger = out.get("trigger")
    if isinstance(trigger, dict):
        trigger = dict(trigger)
        for key in ("source", "mode", "start_time", "end_time"):
            if isinstance(trigger.get(key), str):
                trigger[key] = trigger[key].strip() or None
        if isinstance(trigger.get("mode"), str):
            trigger["mode"] = trigger["mode"].lower()
        last_n = trigger.get("last_n")
        if isinstance(last_n, str) and last_n.strip().isdigit():
            trigger["last_n"] = int(last_n.strip())
        out["trigger"] = trigger

    inp = out.get("input")
    if isinstance(inp, dict):
        inp = dict(inp)
        if isinstance(inp.get("timestamp_column"), str):
            inp["timestamp_column"] = inp["timestamp_column"].strip()
        for key in ("numerical_features", "categorical_features", "targets"):
            value = inp.get(key)
            if isinstance(value, list):
                inp[key] = [v.strip() if isinstance(v, str) else v for v in value]
        out["input"] = inp

    descriptions = out.get("feature_descriptions")
    if isinstance(descriptions, dict):
        out["feature_descriptions"] = {
            (k.strip() if isinstance(k, str) else k):
            (v.strip() if isinstance(v, str) else v)
            for k, v in descriptions.items()
        }
    return out


def _validate_recipe(cfg: Any) -> tuple[list[str], list[str]]:
    """Validate an edited recipe. Returns (errors, warnings).

    Errors block the save; warnings are advisory (returned alongside a
    successful save). Mirrors what _pull_slice / preprocess actually
    require at request time, so a recipe that validates here runs.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(cfg, dict):
        return [f"config payload must be a mapping (got {type(cfg).__name__})"], warnings

    def nonempty_str(v: Any) -> bool:
        return isinstance(v, str) and bool(v.strip())

    system = cfg.get("system")
    if system is not None and not nonempty_str(system):
        errors.append("system must be a non-empty string when provided")

    pipeline = cfg.get("pipeline")
    if not isinstance(pipeline, dict) or not nonempty_str(pipeline.get("output_dir")):
        errors.append("pipeline.output_dir is required (non-empty string)")

    trigger = cfg.get("trigger")
    if not isinstance(trigger, dict):
        errors.append("trigger section is required")
    else:
        if not nonempty_str(trigger.get("source")):
            errors.append("trigger.source is required (path to the shop-floor CSV)")
        mode = trigger.get("mode") or "all"
        if mode not in ("all", "time_range", "last_n"):
            errors.append(
                f"trigger.mode must be one of all / time_range / last_n (got {mode!r})"
            )
        elif mode == "time_range":
            bounds = []
            for label in ("start_time", "end_time"):
                value = trigger.get(label)
                if not nonempty_str(value):
                    errors.append(
                        f"trigger.{label} is required when mode=time_range (ISO 8601)"
                    )
                    bounds.append(None)
                    continue
                try:
                    bounds.append(pd.to_datetime(value, format="ISO8601"))
                except (ValueError, TypeError):
                    errors.append(
                        f"trigger.{label} is not a valid ISO 8601 timestamp: {value!r}"
                    )
                    bounds.append(None)
            if bounds[0] is not None and bounds[1] is not None:
                try:
                    if bounds[0] >= bounds[1]:
                        errors.append("trigger.start_time must be earlier than trigger.end_time")
                except TypeError:
                    errors.append(
                        "trigger.start_time and end_time must both include "
                        "or both omit a timezone"
                    )
        elif mode == "last_n":
            n = trigger.get("last_n")
            if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
                errors.append("trigger.last_n must be a positive integer when mode=last_n")

    numerical: list[str] = []
    categorical: list[str] = []
    inp = cfg.get("input")
    if not isinstance(inp, dict):
        errors.append("input section is required")
    else:
        timestamp = inp.get("timestamp_column")
        if not nonempty_str(timestamp):
            errors.append("input.timestamp_column is required (non-empty string)")

        def str_list(key: str, *, required: bool = False) -> list[str]:
            value = inp.get(key) or []
            if not isinstance(value, list):
                errors.append(f"input.{key} must be a list")
                return []
            if any(not nonempty_str(v) for v in value):
                errors.append(f"input.{key} contains empty or non-string entries")
            valid = [v for v in value if nonempty_str(v)]
            if required and not valid:
                errors.append(f"input.{key} must contain at least one column")
            return valid

        numerical = str_list("numerical_features")
        categorical = str_list("categorical_features")
        targets = str_list("targets", required=True)
        if not numerical and not categorical:
            errors.append(
                "at least one feature is required across "
                "numerical_features + categorical_features"
            )

        all_columns = (
            ([timestamp] if nonempty_str(timestamp) else [])
            + numerical + categorical + targets
        )
        seen: set[str] = set()
        dupes: set[str] = set()
        for col in all_columns:
            if col in seen:
                dupes.add(col)
            seen.add(col)
        if dupes:
            errors.append(
                "duplicate column(s) across timestamp/features/targets: "
                + ", ".join(sorted(dupes))
            )

    descriptions = cfg.get("feature_descriptions")
    if descriptions is not None and not isinstance(descriptions, dict):
        errors.append("feature_descriptions must be a mapping of feature -> description")
    elif isinstance(descriptions, dict):
        empty = sorted(str(k) for k, v in descriptions.items() if not nonempty_str(v))
        if empty:
            errors.append(
                "feature_descriptions has empty descriptions for: " + ", ".join(empty)
            )
        features = set(numerical) | set(categorical)
        if features:
            unknown = sorted(set(descriptions) - features)
            if unknown:
                warnings.append(
                    "descriptions refer to features not in the schema: "
                    + ", ".join(unknown)
                )
            missing = sorted(features - set(descriptions))
            if missing:
                warnings.append(
                    "features without a description (the LLM explains better with one): "
                    + ", ".join(missing)
                )

    return errors, warnings


@app.get("/configs", tags=["config"])
def list_configs() -> dict:
    """List the config recipes available under machines/."""
    configs = []
    if MACHINES_ROOT.is_dir():
        for p in sorted(MACHINES_ROOT.glob("*/*.y*ml")):
            if not p.is_file():
                continue
            configs.append({
                "machine": p.parent.name,
                "name": p.name,
                "recipe": f"{p.parent.name}/{p.name}",
                # The string /pipeline/run and /pipeline/run_csv accept.
                "config_path": f"machines/{p.parent.name}/{p.name}",
            })
    return {"configs": configs}


@app.get("/configs/load", tags=["config"])
def load_config_recipe(path: str) -> dict:
    """Load a recipe as structured data for the dashboard editor."""
    p = _safe_recipe_path(path)
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid YAML in {p.name}: {exc}"
        ) from exc
    if not isinstance(cfg, dict):
        raise HTTPException(
            status_code=400, detail=f"{p.name} does not contain a YAML mapping"
        )
    return {
        "recipe": f"{p.parent.name}/{p.name}",
        "config_path": f"machines/{p.parent.name}/{p.name}",
        "config": cfg,
    }


@app.post("/configs/validate", tags=["config"])
def validate_config_recipe(req: SaveConfigRequest) -> dict:
    """Dry-run validation: same checks as /configs/save, nothing written."""
    _safe_recipe_path(req.path)
    errors, warnings = _validate_recipe(_normalize_recipe(req.config))
    return {
        "status": "ok" if not errors else "invalid",
        "errors": errors,
        "warnings": warnings,
    }


@app.post("/configs/save", tags=["config"])
def save_config_recipe(req: SaveConfigRequest) -> dict:
    """Validate and persist an edited recipe.

    Returns status='invalid' with the error list when validation fails
    (nothing is written). On success the YAML is regenerated with
    yaml.safe_dump — user input is never spliced into the file as text —
    and the previous version is backed up to machines/<m>/.backups/.
    """
    src = _safe_recipe_path(req.path)
    cfg = _normalize_recipe(req.config)
    errors, warnings = _validate_recipe(cfg)
    if errors:
        return {"status": "invalid", "errors": errors, "warnings": warnings}

    dest = src
    if req.save_as:
        name = req.save_as.strip().removesuffix(".yaml").removesuffix(".yml")
        if not _RECIPE_NAME_RE.fullmatch(name):
            raise HTTPException(
                status_code=400,
                detail=(
                    "save_as must be 1-64 characters: letters, digits, "
                    "'_' or '-' (no slashes or dots)"
                ),
            )
        dest = src.parent / f"{name}.yaml"

    # Start from the on-disk YAML so unknown top-level keys an engineer
    # added by hand survive a dashboard save.
    base: dict = {}
    base_src = dest if dest.is_file() else src
    try:
        loaded = yaml.safe_load(base_src.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            base = loaded
    except yaml.YAMLError:
        base = {}  # corrupt file on disk -> the editor payload becomes the new truth
    for section in _EDITABLE_SECTIONS:
        if section in cfg:
            base[section] = cfg[section]

    backup_name = None
    if dest.is_file():
        backup_dir = dest.parent / ".backups"
        backup_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = backup_dir / f"{dest.stem}.{stamp}.yaml"
        shutil.copy2(dest, backup_path)
        backup_name = f"{dest.parent.name}/.backups/{backup_path.name}"

    header = (
        "# Saved by the PCIL dashboard config editor on "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}.\n"
        "# Hand-written YAML comments are not preserved by editor saves.\n"
    )
    dest.write_text(
        header + yaml.safe_dump(base, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    return {
        "status": "ok",
        "recipe": f"{dest.parent.name}/{dest.name}",
        "config_path": f"machines/{dest.parent.name}/{dest.name}",
        "backup": backup_name,
        "warnings": warnings,
    }


class CreateConfigRequest(BaseModel):
    machine: str = Field(
        ...,
        description="New machine folder name (letters, digits, '_', '-').",
        examples=["laser_welder"],
    )
    name: str = Field(
        "config",
        description="Recipe filename without extension (defaults to 'config').",
    )
    config: dict = Field(
        ...,
        description="The initial recipe — same shape /configs/save validates.",
    )


@app.post("/configs/create", tags=["config"])
def create_config_recipe(req: CreateConfigRequest) -> dict:
    """Create a brand-new machine folder + recipe from the dashboard.

    Same validation as /configs/save; refuses to overwrite an existing
    recipe (use /configs/save for edits). The machine's output/ folder
    is created automatically on the first pipeline run.
    """
    machine = req.machine.strip()
    name = req.name.strip().removesuffix(".yaml").removesuffix(".yml") or "config"
    if not _RECIPE_NAME_RE.fullmatch(machine):
        raise HTTPException(
            status_code=400,
            detail=(
                "machine must be 1-64 characters: letters, digits, '_' or '-' "
                "(no slashes, dots or spaces)"
            ),
        )
    if not _RECIPE_NAME_RE.fullmatch(name):
        raise HTTPException(
            status_code=400,
            detail="name must be 1-64 characters: letters, digits, '_' or '-'",
        )

    cfg = _normalize_recipe(req.config)
    errors, warnings = _validate_recipe(cfg)
    if errors:
        return {"status": "invalid", "errors": errors, "warnings": warnings}

    dest = MACHINES_ROOT / machine / f"{name}.yaml"
    if dest.is_file():
        raise HTTPException(
            status_code=409,
            detail=(
                f"recipe {machine}/{name}.yaml already exists — "
                "edit it via /configs/save instead"
            ),
        )

    payload = {section: cfg[section] for section in _EDITABLE_SECTIONS if section in cfg}
    header = (
        "# Created by the PCIL dashboard config editor on "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}.\n"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        header + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    return {
        "status": "ok",
        "recipe": f"{machine}/{name}.yaml",
        "config_path": f"machines/{machine}/{name}.yaml",
        "warnings": warnings,
    }


class DeleteConfigRequest(BaseModel):
    path: str = Field(
        ...,
        description="Recipe to delete, e.g. 'inkjet_printer/config.yaml'.",
    )


@app.post("/configs/delete", tags=["config"])
def delete_config_recipe(req: DeleteConfigRequest) -> dict:
    """Delete a recipe — recoverably.

    The file is MOVED to machines/<machine>/.backups/<name>.deleted-<stamp>.yaml
    rather than destroyed, so a wrong click can be undone from disk. A
    machine whose last recipe is deleted simply disappears from /configs;
    its folder (with backups and outputs) stays on disk.
    """
    p = _safe_recipe_path(req.path)
    backup_dir = p.parent / ".backups"
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"{p.stem}.deleted-{stamp}.yaml"
    shutil.move(str(p), str(backup_path))
    return {
        "status": "ok",
        "deleted": f"{p.parent.name}/{p.name}",
        "backup": f"{p.parent.name}/.backups/{backup_path.name}",
    }


# Bundle filenames follow <model_type>_<model_id>.pkl; model_type itself
# contains an underscore for non_cyclical, so match the known prefixes.
_BUNDLE_STEM_RE = re.compile(r"^(non_cyclical|cyclical|irregular)_(.+)$")


@app.get("/anomaly/models", tags=["anomaly"])
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

    scored = non_cyclical_score(df, bundle)

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

    scored = cyclical_score(df, bundle)

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
    except ValueError as exc:
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


@app.post("/anomaly/train", tags=["anomaly"])
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

    elif model_type == "irregular" and training_mode == "normal_only":
        if file is None:
            raise HTTPException(
                status_code=400,
                detail=("irregular normal_only requires the 'file' form field "
                        "(the training CSV)."),
            )
        df = await _read_upload_to_df(file)
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

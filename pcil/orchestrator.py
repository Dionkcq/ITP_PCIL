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

    # --- RAG retrieval + LLM composition ---------------------------
    # Retrieval and composition are guarded separately: retrieval is
    # fully local (DOCX files on disk), composition needs the Gemini
    # API. If only composition fails (no key, no internet, timeout),
    # the operator still gets the retrieved recovery records.
    if RAG_DIR.is_dir():
        try:
            rag_query = _build_rag_query(impacts)
            all_records = load_all_recovery_docs(RAG_DIR)
            recovery_records = lookup_keywords(rag_query, all_records, top_k=3)
        except Exception as exc:  # noqa: BLE001
            recovery_records = []
            operator_recommendation = (
                f"RAG retrieval failed ({type(exc).__name__}): {exc}. "
                "Review the impacts data manually."
            )
        else:
            try:
                operator_recommendation = compose_recommendation(
                    impacts, recovery_records,
                )
            except Exception as exc:  # noqa: BLE001
                operator_recommendation = (
                    f"LLM composition failed ({type(exc).__name__}): {exc}. "
                    "Review the recovery records below manually."
                )
    else:
        recovery_records = []
        operator_recommendation = (
            "RAG document directory not found. "
            f"Expected: {RAG_DIR}. "
            "Mount the data/RAG/ folder and restart the orchestrator."
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
        "recovery_records": recovery_records,
        "operator_recommendation": operator_recommendation,
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


def _build_rag_query(impacts: dict) -> str:
    """Build a keyword query string from the impacts dict for RAG retrieval.

    Extracts vocabulary from feature descriptions so that tokens match
    human-readable DOCX error text. Falls back to splitting the column name
    on underscores when no description is available.
    """
    tokens: set[str] = set()
    for block in impacts.get("context", []):
        tokens.add(block["target"])
        for fi in block.get("ranked_feature_impacts", [])[:2]:
            description = fi.get("description", "")
            if description:
                tokens.update(description.lower().split())
            else:
                tokens.update(fi["feature"].lower().split("_"))
    return " ".join(tokens)


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
            "config": [
                "GET /configs",
                "GET /configs/load",
                "POST /configs/validate",
                "POST /configs/save",
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
        "is_anomaly": (
            scored["is_anomaly"].tolist()
            if "is_anomaly" in scored.columns
            else None
        ),
        "threshold": bundle.get("threshold"),
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

    return {
        "status": "ok",
        "model_type": "irregular",
        "model_id": model_id,
        "input_rows": int(len(df)),
        "windows_scored": int(len(scored)),
        "anomaly_scores": scored["anomaly_score"].tolist(),
        "is_anomaly": scored["is_anomaly"].tolist(),
        "threshold": bundle.get("threshold"),
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

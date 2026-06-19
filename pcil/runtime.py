"""Shared runtime helpers for the PCIL services.

Used by BOTH the pipeline app (`orchestrator.py`) and the standalone anomaly
app (`anomaly_app.py`). Kept dependency-light (no torch, no RAG, no DB) so the
lightweight pipeline image can import it without pulling the anomaly stack, and
the anomaly image can import it without pulling the pipeline/RAG/DB stack.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
from fastapi import HTTPException, UploadFile


# Project root for resolving relative data paths (anomaly bundles, the mock CSV,
# the RAG dir). In dev: pcil/runtime.py -> pcil/ -> PCIL_dev/ -> ITP/ is the root
# (parents[2]); in Docker /app is the root, set via PCIL_PROJECT_ROOT=/app.
PROJECT_ROOT = Path(
    os.environ.get("PCIL_PROJECT_ROOT") or Path(__file__).resolve().parents[2]
)


def service() -> str:
    """Which service this process runs as: full | pipeline | anomaly.

    full     - single container, every endpoint (default; the :postgres /
               :latest image and the current single-container deployment).
    pipeline - pipeline + config + shopfloor + RAG + dashboard, NO anomaly
               (the lightweight, torch-free image); /anomaly/* is proxied to the
               anomaly service when ANOMALY_SERVICE_URL is set.
    anomaly  - only the /anomaly/* API (the torch image; pcil/anomaly_app.py).
    """
    return os.environ.get("PCIL_SERVICE", "full").lower()


def bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


async def read_upload_to_df(upload: UploadFile, *, skiprows: int = 0) -> pd.DataFrame:
    """Read a FastAPI UploadFile into a pandas DataFrame.

    `skiprows` drops that many leading lines before the header row. Set it to
    5 for raw WebDAQ acoustic exports, whose CSV opens with a device-info
    preamble (device name, sample rate, start time, blank line) before the
    real "Sample,Time (s),Acceleration ..." header. Defaults to 0 (the header
    is the first line), which keeps already-clean uploads unchanged.

    Raises HTTPException(400) if the file is empty, can't be parsed as CSV, or
    parses to an empty DataFrame.
    """
    contents = await upload.read()
    if not contents:
        raise HTTPException(
            status_code=400,
            detail=f"uploaded file '{upload.filename}' is empty",
        )
    try:
        df = pd.read_csv(io.BytesIO(contents), skiprows=skiprows)
    except Exception as exc:  # noqa: BLE001 - surface any pandas parse error
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

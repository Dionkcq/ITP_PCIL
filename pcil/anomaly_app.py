"""Standalone PCIL Anomaly service (PCIL_SERVICE=anomaly).

A thin FastAPI app that mounts ONLY the /anomaly/* router. It deliberately does
NOT import the orchestrator, so the anomaly image needs only the anomaly stack
(torch + scikit-learn + scipy + pandas + joblib) - not the pipeline / RAG / DB /
dashboard dependencies.

Run:
    uvicorn pcil.anomaly_app:app --host 0.0.0.0 --port 8000

This is the engineer-facing "score / train" service from Winardi's 22 May design
(input -> output only): some projects deploy it alone, without the pipeline.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pcil.anomaly_api import router

app = FastAPI(
    title="PCIL Anomaly Service",
    version="0.1.0",
    description=(
        "Anomaly scoring + training API (input -> output only). "
        "See /docs for the interactive Swagger UI."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["meta"])
def root() -> dict:
    """Service metadata + endpoint index."""
    return {
        "service": "PCIL Anomaly Service",
        "version": app.version,
        "endpoints": {
            "anomaly": [
                "POST /anomaly/train",
                "POST /anomaly/score",
                "GET /anomaly/models",
            ],
            "docs": "GET /docs",
        },
    }

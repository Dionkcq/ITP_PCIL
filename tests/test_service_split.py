"""Tests for the P2 service split (PCIL_SERVICE: full | pipeline | anomaly).

- The standalone anomaly app (pcil.anomaly_app) serves only /anomaly/*.
- In pipeline mode the orchestrator exposes /anomaly/* as a proxy that 503s
  when ANOMALY_SERVICE_URL is unset (rather than 404, so the dashboard can show
  a clear message).
- Full mode (the default everywhere else in the suite) keeps every endpoint.
"""

from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_service_helper_reads_env(monkeypatch):
    import pcil.runtime as runtime

    monkeypatch.delenv("PCIL_SERVICE", raising=False)
    assert runtime.service() == "full"
    monkeypatch.setenv("PCIL_SERVICE", "PIPELINE")
    assert runtime.service() == "pipeline"


def test_anomaly_app_serves_metadata_and_models(isolated_data_dir):
    import pcil.anomaly_app as anomaly_app

    client = TestClient(anomaly_app.app)
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "PCIL Anomaly Service"

    # /anomaly/models reads anomaly_api.PROJECT_ROOT, which isolated_data_dir
    # redirects to the tmp data dir.
    (isolated_data_dir / "cyclical_inkjet_01.pkl").write_bytes(b"x")
    r = client.get("/anomaly/models")
    assert r.status_code == 200
    assert any(m["model_type"] == "cyclical" for m in r.json()["models"])


def test_pipeline_mode_proxies_anomaly_with_503_when_unconfigured(monkeypatch):
    """Reload the orchestrator as the 'pipeline' service and confirm /anomaly/*
    is the proxy (503 without ANOMALY_SERVICE_URL), then restore full mode."""
    monkeypatch.setenv("PCIL_SERVICE", "pipeline")
    monkeypatch.delenv("ANOMALY_SERVICE_URL", raising=False)
    import pcil.orchestrator as orch

    try:
        importlib.reload(orch)
        client = TestClient(orch.app)
        r = client.get("/anomaly/models")
        assert r.status_code == 503
        assert "anomaly" in r.json()["detail"].lower()
    finally:
        # Restore the default (full) app so later tests see every endpoint.
        monkeypatch.delenv("PCIL_SERVICE", raising=False)
        importlib.reload(orch)

"""Tests for the service metadata endpoint GET /."""


def test_root_returns_service_info(client):
    """GET / responds 200 and identifies the service."""
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "PCIL Job Orchestrator"
    assert "version" in body


def test_root_lists_all_factory_endpoints(client):
    """The endpoint index should include every factory-test endpoint
    so engineers reading the response know what's available."""
    r = client.get("/")
    body = r.json()
    pipeline = body["endpoints"]["pipeline"]
    anomaly = body["endpoints"]["anomaly"]

    assert "POST /pipeline/run" in pipeline
    assert "POST /pipeline/run_csv" in pipeline
    assert "POST /pipeline/save_csv" in pipeline
    assert "POST /anomaly/train" in anomaly
    assert "POST /anomaly/score" in anomaly
    assert body["endpoints"]["docs"] == "GET /docs"

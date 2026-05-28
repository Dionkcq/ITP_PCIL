"""Tests for POST /anomaly/score.

Covers the input validation + 404 paths. Happy-path scoring against
real trained bundles is exercised by `scripts/smoke_test_orchestrator.py`
(which depends on bundles being present in data/), so it isn't repeated
here to keep `pytest` fast and hermetic.
"""


def test_score_rejects_empty_data_cyclical(client):
    """The endpoint should reject an empty 'data' list with 400, BEFORE
    trying to load any bundle."""
    r = client.post(
        "/anomaly/score",
        json={"data": [], "model_type": "cyclical", "model_id": "anything"},
    )
    assert r.status_code == 400
    assert "data" in r.json()["detail"]


def test_score_rejects_empty_data_non_cyclical(client):
    r = client.post(
        "/anomaly/score",
        json={"data": [], "model_type": "non_cyclical", "model_id": "anything"},
    )
    assert r.status_code == 400


def test_score_returns_404_for_missing_cyclical_bundle(client, isolated_data_dir):
    """When the bundle doesn't exist on disk, the endpoint returns 404
    with a clear message telling the engineer how to train one.

    Uses isolated_data_dir so we're guaranteed the bundle is missing
    regardless of what's in the real ITP/data/ folder."""
    r = client.post(
        "/anomaly/score",
        json={
            "data": [{"timestamp": "2026-06-12T09:00:00", "machine_id": "x", "signal_value": 0.1}],
            "model_type": "cyclical",
            "model_id": "definitely_not_trained",
        },
    )
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert "cyclical bundle not found" in detail
    assert "/anomaly/train" in detail  # message should point to the training endpoint


def test_score_returns_404_for_missing_non_cyclical_bundle(client, isolated_data_dir):
    r = client.post(
        "/anomaly/score",
        json={
            "data": [{"Acceleration 0 (g)": 0.0, "Acceleration 1 (g)": 0.0,
                      "Acceleration 2 (g)": 0.0, "AE (V) (V)": 0.0}],
            "model_type": "non_cyclical",
            "model_id": "definitely_not_trained",
        },
    )
    assert r.status_code == 404
    assert "non_cyclical bundle not found" in r.json()["detail"]


def test_score_rejects_unknown_model_type(client):
    """Pydantic Literal type bound should reject anything that isn't
    'cyclical' or 'non_cyclical' at request-parse time (422)."""
    r = client.post(
        "/anomaly/score",
        json={
            "data": [{"x": 1}],
            "model_type": "not_a_real_type",
            "model_id": "x",
        },
    )
    assert r.status_code == 422

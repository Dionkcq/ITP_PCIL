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


# ─────────────────────────────────────────────────────────────
# Zero-cycles / zero-windows regression tests
#
# Input that is valid but too SHORT to produce a single cycle/window must
# return "0 scored" with empty arrays — not a 500. Found live (2026-06-11)
# when scoring the 50-row cyclical fixture against the real CNN bundle:
# stack_features([]) produced a column-less DataFrame and the normaliser's
# groupby raised KeyError('machine_id').
#
# The bundles below carry the real key layout, but normaliser/model are
# None — the scorer must return BEFORE touching them when no cycles or
# windows are detected.
# ─────────────────────────────────────────────────────────────

def test_score_cyclical_short_input_returns_zero_cycles(
    client, isolated_data_dir,
):
    import joblib

    bundle = {
        "machine_id_column": "machine_id",
        "signal_column": "signal_value",
        "timestamp_column": "timestamp",
        "feature_columns": ["f0", "f1"],
        "normaliser": None,
        "model": None,
        "threshold": 0.5,
    }
    joblib.dump(bundle, isolated_data_dir / "cyclical_short_test.pkl")

    rows = [
        {"timestamp": f"2026-06-12T09:00:{i:02d}", "machine_id": "inkjet_01",
         "signal_value": 0.1 * i}
        for i in range(10)
    ]
    r = client.post(
        "/anomaly/score",
        json={"data": rows, "model_type": "cyclical", "model_id": "short_test"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["input_rows"] == 10
    assert body["cycles_scored"] == 0
    assert body["anomaly_scores"] == []
    assert body["is_anomaly"] == []
    assert body["threshold"] == 0.5
    assert body["threshold_source"] == "bundle_95th_percentile"


def test_score_non_cyclical_short_input_returns_zero_windows(
    client, isolated_data_dir,
):
    import joblib

    channels = [
        "Acceleration 0 (g)", "Acceleration 1 (g)",
        "Acceleration 2 (g)", "AE (V) (V)",
    ]
    bundle = {
        "machine_id": "inkjet_01",
        "machine_id_column": "machine_id",
        "window_size_rows": 12800,   # far more rows than we send
        "channel_columns": channels,
        "feature_columns": ["f0", "f1"],
        "normaliser": None,
        "model": None,
    }
    joblib.dump(bundle, isolated_data_dir / "non_cyclical_short_test.pkl")

    rows = [{c: 0.01 * i for c in channels} for i in range(3)]
    r = client.post(
        "/anomaly/score",
        json={"data": rows, "model_type": "non_cyclical", "model_id": "short_test"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["input_rows"] == 3
    assert body["windows_scored"] == 0
    assert body["anomaly_scores"] == []
    assert body["is_anomaly"] is None
    assert body["threshold"] is None
    assert body["threshold_source"] == "not_configured"
    assert body["window_starts"] == []

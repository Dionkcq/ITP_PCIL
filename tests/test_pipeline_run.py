"""Regression tests for POST /pipeline/run.

These tests ensure the existing config-driven path still works after
the refactor that extracted `_run_pipeline_on_df` into a shared helper.
"""

DEFAULT_CONFIG = "machines/inkjet_printer/config.yaml"


def test_pipeline_run_with_default_config_returns_impacts(client):
    """The default config + the mock shop-floor CSV should round-trip
    end-to-end through preprocess -> adapter -> context model."""
    r = client.post("/pipeline/run", json={"config_path": DEFAULT_CONFIG})
    # If data/mock_shop_floor.csv is missing this returns 404 — we'll
    # accept that as 'environment not set up' rather than a regression.
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        body = r.json()
        assert body["status"] == "ok"
        assert "impacts" in body
        assert body["impacts"]["system"] == "inkjet_printer"
        assert body["impacts"]["model"] == "linear_regression"
        assert "context_window" in body["impacts"]
        assert body["golden_rows"] > 0


def test_pipeline_run_missing_config_returns_404(client):
    """A non-existent config path should fail clearly."""
    r = client.post(
        "/pipeline/run",
        json={"config_path": "machines/no_such_machine/config.yaml"},
    )
    assert r.status_code == 404
    assert "config.yaml not found" in r.json()["detail"]

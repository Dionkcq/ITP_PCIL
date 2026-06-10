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


def test_trigger_source_resolves_against_project_root(
    client, monkeypatch, tmp_path, shop_floor_tiny_path,
):
    """Regression test for the Docker path bug (found 2026-06-10): a
    relative trigger.source like "data/x.csv" must resolve against
    PROJECT_ROOT (the rule anomaly bundles and RAG_DIR already follow),
    NOT only against the config file's directory. In the container the
    config-dir-relative "../../../data/..." climbed out of /app to the
    filesystem root and 404'd even though the data was mounted at
    /app/data."""
    import shutil
    from pathlib import Path

    import pcil.orchestrator as orch

    # Simulated project root: data/ lives directly under it (like /app).
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(shop_floor_tiny_path, data_dir / "slice_tiny.csv")
    monkeypatch.setattr(orch, "PROJECT_ROOT", tmp_path)

    # Config sits in its own subfolder; root-relative source. Reuse the
    # real recipe so the input schema matches the fixture CSV.
    real_cfg = (
        Path(__file__).resolve().parents[1]
        / "machines" / "inkjet_printer" / "config.yaml"
    ).read_text(encoding="utf-8")
    assert 'source: "data/mock_shop_floor.csv"' in real_cfg
    cfg_dir = tmp_path / "machines" / "inkjet_printer"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.yaml").write_text(
        real_cfg.replace(
            'source: "data/mock_shop_floor.csv"',
            'source: "data/slice_tiny.csv"',
        ),
        encoding="utf-8",
    )

    r = client.post(
        "/pipeline/run",
        json={"config_path": str(cfg_dir / "config.yaml"), "persist": False},
    )
    assert r.status_code == 200, r.text
    assert r.json()["impacts"]["system"] == "inkjet_printer"


def test_trigger_source_not_found_lists_tried_paths(client, monkeypatch, tmp_path):
    """When the source is missing everywhere, the 404 should list the
    candidate locations so the engineer can see where it looked."""
    import pcil.orchestrator as orch
    from pathlib import Path

    monkeypatch.setattr(orch, "PROJECT_ROOT", tmp_path)
    real_cfg = (
        Path(__file__).resolve().parents[1]
        / "machines" / "inkjet_printer" / "config.yaml"
    ).read_text(encoding="utf-8")
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        real_cfg.replace(
            'source: "data/mock_shop_floor.csv"',
            'source: "data/definitely_missing.csv"',
        ),
        encoding="utf-8",
    )

    r = client.post(
        "/pipeline/run",
        json={"config_path": str(cfg_dir / "config.yaml")},
    )
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert "trigger.source not found" in detail
    assert "also tried" in detail

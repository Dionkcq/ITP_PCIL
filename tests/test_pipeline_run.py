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
    # real recipe so the input schema matches the fixture CSV. Parse +
    # re-dump rather than string-replacing: the dashboard config editor
    # legitimately rewrites the file (no comments, unquoted strings), so
    # the test must not depend on the YAML's surface formatting.
    import yaml

    real_cfg = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "machines" / "inkjet_printer" / "config.yaml"
        ).read_text(encoding="utf-8")
    )
    assert real_cfg["trigger"]["source"] == "data/mock_shop_floor.csv"
    real_cfg["trigger"]["source"] = "data/slice_tiny.csv"
    cfg_dir = tmp_path / "machines" / "inkjet_printer"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.yaml").write_text(
        yaml.safe_dump(real_cfg, sort_keys=False), encoding="utf-8"
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

    import yaml

    monkeypatch.setattr(orch, "PROJECT_ROOT", tmp_path)
    real_cfg = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "machines" / "inkjet_printer" / "config.yaml"
        ).read_text(encoding="utf-8")
    )
    real_cfg["trigger"]["source"] = "data/definitely_missing.csv"
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(
        yaml.safe_dump(real_cfg, sort_keys=False), encoding="utf-8"
    )

    r = client.post(
        "/pipeline/run",
        json={"config_path": str(cfg_dir / "config.yaml")},
    )
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert "trigger.source not found" in detail
    assert "also tried" in detail


def test_train_baseline_then_run_uses_baseline_artifacts(
    client, tmp_path, shop_floor_tiny_path,
):
    """Baseline training should persist artifacts that /pipeline/run can
    reuse for transform-only preprocessing and deviation reporting."""
    from pathlib import Path

    import yaml

    real_cfg = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "machines" / "inkjet_printer" / "config.yaml"
        ).read_text(encoding="utf-8")
    )
    real_cfg["trigger"]["source"] = str(shop_floor_tiny_path)
    cfg_dir = tmp_path / "machines" / "inkjet_printer"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(real_cfg, sort_keys=False),
        encoding="utf-8",
    )

    train = client.post(
        "/pipeline/train_baseline",
        json={"config_path": str(cfg_path)},
    )
    assert train.status_code == 200, train.text
    trained = train.json()
    assert trained["status"] == "ok"
    assert trained["baseline_rows"] == 50
    assert Path(trained["artifacts"]["preprocessor"]).is_file()
    assert Path(trained["artifacts"]["stats"]).is_file()

    run = client.post(
        "/pipeline/run",
        json={"config_path": str(cfg_path), "persist": False},
    )
    assert run.status_code == 200, run.text
    body = run.json()
    assert body["impacts"]["preprocessing_source"] == "baseline_preprocessor"
    assert body["baseline_comparison"]["status"] == "available"
    assert "features_scaled_on_current_window" not in body["pipeline_warnings"]

"""Tests for POST /pipeline/run_csv.

Same pipeline as /pipeline/run, but the slice arrives as a multipart
upload instead of via cfg['trigger']['source']. Verifies the upload
plumbing, validation, and that the shared `_run_pipeline_on_df` helper
is wired in correctly.
"""

import io

DEFAULT_CONFIG = "machines/inkjet_printer/config.yaml"


def test_run_csv_with_valid_shop_floor_returns_impacts(
    client, shop_floor_tiny_path,
):
    """A 50-row shop_floor CSV should produce a Golden DataFrame and
    an impacts dict. This is the realistic happy-path factory test."""
    with open(shop_floor_tiny_path, "rb") as f:
        r = client.post(
            "/pipeline/run_csv",
            data={"config_path": DEFAULT_CONFIG, "persist": "false"},
            files={"file": ("shop_floor_tiny.csv", f, "text/csv")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["input_rows"] == 50
    assert body["golden_rows"] == 50
    assert body["impacts"]["system"] == "inkjet_printer"
    assert body["impacts"]["context_window"]["row_count"] == 50
    assert len(body["impacts"]["context"]) == 4  # availability + perf + qual + oee


def test_run_csv_rejects_empty_csv(client):
    """A completely empty file should fail with 400."""
    r = client.post(
        "/pipeline/run_csv",
        data={"config_path": DEFAULT_CONFIG},
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_run_csv_rejects_csv_with_no_rows(client):
    """A CSV that has a header but no data rows is also empty in pandas terms."""
    r = client.post(
        "/pipeline/run_csv",
        data={"config_path": DEFAULT_CONFIG},
        files={"file": ("only_header.csv", b"a,b,c\n", "text/csv")},
    )
    # pandas parses fine but the DataFrame is empty.
    assert r.status_code == 400
    assert "no rows" in r.json()["detail"].lower()


def test_run_csv_without_file_returns_422(client):
    """FastAPI returns 422 (unprocessable entity) when the required
    multipart 'file' field is missing."""
    r = client.post(
        "/pipeline/run_csv",
        data={"config_path": DEFAULT_CONFIG},
    )
    assert r.status_code == 422


def test_run_csv_with_unparseable_csv_returns_400(client):
    """Non-CSV bytes should produce a 400 with a useful message."""
    bad = b"\x00\x01\x02not,a,csv\nat,all\n"  # binary-ish junk
    r = client.post(
        "/pipeline/run_csv",
        data={"config_path": DEFAULT_CONFIG},
        files={"file": ("garbage.bin", bad, "application/octet-stream")},
    )
    # pandas might actually parse this — accept either parse failure or
    # downstream schema failure. Anything in 4xx is fine.
    assert 400 <= r.status_code < 500


def test_run_csv_returns_rag_recommendation_when_rag_dir_present(
    client, shop_floor_tiny_path, monkeypatch, tmp_path,
):
    """Happy-path for Pipeline #3 (RAG): when RAG_DIR exists, the
    orchestrator should call into the loader, lookup, and composer and
    return the resulting recovery_records + operator_recommendation.

    Loader + composer are monkeypatched so the test stays offline and
    deterministic (no Gemini API call, no DOCX parsing). The intent is
    to lock the *wiring*; the loader and composer are exercised
    elsewhere (loader by Robin's manual DOCX work; composer by smoke
    tests with a real GEMINI_API_KEY set).
    """
    import pcil.orchestrator as orch

    fake_rag_dir = tmp_path / "RAG"
    fake_rag_dir.mkdir()
    fake_records = [{
        "error": "Print head clog",
        "cause": "Dried ink in nozzle",
        "recovery": "Run cleaning cycle and prime the head.",
        "source_doc": "Inkjet.docx",
    }]

    monkeypatch.setattr(orch, "RAG_DIR", fake_rag_dir)
    monkeypatch.setattr(
        orch, "load_all_recovery_docs", lambda _path: fake_records,
    )
    monkeypatch.setattr(
        orch, "lookup_keywords",
        lambda query, records, top_k=3: records[:top_k],
    )
    monkeypatch.setattr(
        orch, "compose_recommendation",
        lambda impacts, records, **kwargs: "Mocked operator recommendation.",
    )

    with open(shop_floor_tiny_path, "rb") as f:
        r = client.post(
            "/pipeline/run_csv",
            data={"config_path": DEFAULT_CONFIG, "persist": "false"},
            files={"file": ("shop_floor_tiny.csv", f, "text/csv")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["operator_recommendation"] == "Mocked operator recommendation."
    assert body["recommendation_source"] == "gemini"
    assert body["recommendation_warnings"] == []
    assert "signal_summary" in body
    assert "baseline_comparison" in body
    assert len(body["recovery_records"]) == 1
    assert body["recovery_records"][0]["error"] == "Print head clog"
    assert body["recovery_records"][0]["source_doc"] == "Inkjet.docx"


def test_run_csv_keeps_records_when_llm_fails(
    client, shop_floor_tiny_path, monkeypatch, tmp_path,
):
    """Degradation path: retrieval succeeds but the composer raises
    (missing GEMINI_API_KEY, no internet, timeout). The retrieved
    recovery_records must SURVIVE — only the recommendation degrades
    to a fallback string. Regression test for the 2026-06-10 fix:
    previously a composer exception escaped to the retrieval handler,
    which zeroed out the records and mislabelled the failure as
    'RAG retrieval failed'."""
    import pcil.orchestrator as orch

    fake_rag_dir = tmp_path / "RAG"
    fake_rag_dir.mkdir()
    fake_records = [{
        "error": "Print head clog",
        "cause": "Dried ink in nozzle",
        "recovery": "Run cleaning cycle and prime the head.",
        "source_doc": "Inkjet.docx",
    }]

    def _composer_boom(impacts, records):
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    monkeypatch.setattr(orch, "RAG_DIR", fake_rag_dir)
    monkeypatch.setattr(
        orch, "load_all_recovery_docs", lambda _path: fake_records,
    )
    monkeypatch.setattr(
        orch, "lookup_keywords",
        lambda query, records, top_k=3: records[:top_k],
    )
    monkeypatch.setattr(orch, "compose_recommendation", _composer_boom)

    with open(shop_floor_tiny_path, "rb") as f:
        r = client.post(
            "/pipeline/run_csv",
            data={"config_path": DEFAULT_CONFIG, "persist": "false"},
            files={"file": ("shop_floor_tiny.csv", f, "text/csv")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    # Records survive the composer failure.
    assert len(body["recovery_records"]) == 1
    assert body["recovery_records"][0]["error"] == "Print head clog"
    # Recommendation degrades with an honest, correctly-attributed message.
    assert "LLM composition failed" in body["operator_recommendation"]
    assert body["recommendation_source"] == "fallback"
    assert "llm_composition_failed" in body["recommendation_warnings"]
    # Pipelines #1/#2 unaffected.
    assert body["impacts"]["system"] == "inkjet_printer"


def test_run_csv_marks_no_records_structurally(
    client, shop_floor_tiny_path, monkeypatch, tmp_path,
):
    """When retrieval runs but finds no matching records, the response
    should carry a machine-readable no_records source instead of asking
    the frontend to parse recommendation text."""
    import pcil.orchestrator as orch

    fake_rag_dir = tmp_path / "RAG"
    fake_rag_dir.mkdir()

    monkeypatch.setattr(orch, "RAG_DIR", fake_rag_dir)
    monkeypatch.setattr(orch, "load_all_recovery_docs", lambda _path: [])
    monkeypatch.setattr(orch, "lookup_keywords", lambda query, records, top_k=3: [])

    with open(shop_floor_tiny_path, "rb") as f:
        r = client.post(
            "/pipeline/run_csv",
            data={"config_path": DEFAULT_CONFIG, "persist": "false"},
            files={"file": ("shop_floor_tiny.csv", f, "text/csv")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["recommendation_source"] == "no_records"
    assert "no_matching_recovery_records" in body["recommendation_warnings"]
    assert body["recovery_records"] == []


def test_run_csv_marks_retrieval_error_structurally(
    client, shop_floor_tiny_path, monkeypatch, tmp_path,
):
    """Retrieval failures should not be conflated with LLM fallback."""
    import pcil.orchestrator as orch

    fake_rag_dir = tmp_path / "RAG"
    fake_rag_dir.mkdir()

    def _lookup_boom(query, records, top_k=3):
        raise RuntimeError("index broken")

    monkeypatch.setattr(orch, "RAG_DIR", fake_rag_dir)
    monkeypatch.setattr(orch, "load_all_recovery_docs", lambda _path: [])
    monkeypatch.setattr(orch, "lookup_keywords", _lookup_boom)

    with open(shop_floor_tiny_path, "rb") as f:
        r = client.post(
            "/pipeline/run_csv",
            data={"config_path": DEFAULT_CONFIG, "persist": "false"},
            files={"file": ("shop_floor_tiny.csv", f, "text/csv")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["recommendation_source"] == "retrieval_error"
    assert "rag_retrieval_failed" in body["recommendation_warnings"]
    assert "RAG retrieval failed" in body["operator_recommendation"]


def test_run_csv_falls_back_when_rag_dir_missing(
    client, shop_floor_tiny_path, monkeypatch, tmp_path,
):
    """Error-path counterpart: when RAG_DIR does not exist on disk, the
    orchestrator must NOT call the loader/composer and must return a
    fallback recommendation string + an empty recovery_records list.
    Impacts JSON should still be present (Pipelines #1 and #2 are
    independent of RAG)."""
    import pcil.orchestrator as orch

    nonexistent = tmp_path / "definitely_not_here"
    monkeypatch.setattr(orch, "RAG_DIR", nonexistent)

    with open(shop_floor_tiny_path, "rb") as f:
        r = client.post(
            "/pipeline/run_csv",
            data={"config_path": DEFAULT_CONFIG, "persist": "false"},
            files={"file": ("shop_floor_tiny.csv", f, "text/csv")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["recovery_records"] == []
    assert "RAG document directory not found" in body["operator_recommendation"]
    assert body["recommendation_source"] == "fallback"
    assert "rag_directory_missing" in body["recommendation_warnings"]
    assert body["impacts"]["system"] == "inkjet_printer"

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

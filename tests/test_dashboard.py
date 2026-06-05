"""Tests for serving the built React dashboard from FastAPI.

These tests exercise `attach_dashboard()` directly against a fresh
FastAPI app and a synthetic `dist/` directory. They deliberately do
**not** invoke Node / Vite — `npm run build` is not part of the pytest
contract, so the suite runs cleanly in CI environments that only have
the Python toolchain.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pcil.orchestrator import DASHBOARD_URL_PATH, attach_dashboard


SYNTHETIC_INDEX = (
    "<!doctype html><html><head><title>PCIL Dashboard</title></head>"
    "<body><div id='root'>pcil-dashboard-fixture</div></body></html>"
)


@pytest.fixture
def synthetic_dist(tmp_path: Path) -> Path:
    """A tmp `dist/` directory that mimics what `vite build` produces.

    Just the bits StaticFiles needs: an `index.html` at the root plus
    one asset under `assets/` to assert sub-path serving works.
    """
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(SYNTHETIC_INDEX, encoding="utf-8")
    (dist / "assets" / "app.js").write_text(
        "// pcil-dashboard-fixture-asset", encoding="utf-8"
    )
    return dist


def _client_with_dashboard(dist: Path | None) -> tuple[TestClient, Path | None]:
    """Build a minimal app + mount the dashboard if a dist is provided.

    Mirrors what the real orchestrator module does at import time,
    without coupling the test to the live app instance (which is built
    once per pytest process and would carry state across tests).
    """
    test_app = FastAPI()
    resolved = attach_dashboard(test_app, dist) if dist is not None else None
    return TestClient(test_app), resolved


def test_attach_dashboard_returns_none_when_dist_missing(tmp_path: Path) -> None:
    """No dist directory -> mount is skipped and a hint is returned."""
    missing = tmp_path / "does-not-exist"
    resolved = attach_dashboard(FastAPI(), missing)
    assert resolved is None


def test_attach_dashboard_returns_none_when_index_missing(tmp_path: Path) -> None:
    """A dist directory without index.html is treated as no dashboard."""
    dist = tmp_path / "dist"
    dist.mkdir()
    # No index.html on purpose.
    resolved = attach_dashboard(FastAPI(), dist)
    assert resolved is None


def test_dashboard_serves_index_html(synthetic_dist: Path) -> None:
    """GET /dashboard/ returns the bundled index.html."""
    client, resolved = _client_with_dashboard(synthetic_dist)
    assert resolved == synthetic_dist

    r = client.get(f"{DASHBOARD_URL_PATH}/")
    assert r.status_code == 200
    assert "pcil-dashboard-fixture" in r.text
    assert r.headers["content-type"].startswith("text/html")


def test_dashboard_serves_assets_subpath(synthetic_dist: Path) -> None:
    """Vite emits assets under /assets/*; StaticFiles must serve them."""
    client, _ = _client_with_dashboard(synthetic_dist)

    r = client.get(f"{DASHBOARD_URL_PATH}/assets/app.js")
    assert r.status_code == 200
    assert "pcil-dashboard-fixture-asset" in r.text


def test_env_var_overrides_default(monkeypatch, synthetic_dist: Path) -> None:
    """DASHBOARD_DIST_DIR env var picks up the dist when no arg is passed."""
    monkeypatch.setenv("DASHBOARD_DIST_DIR", str(synthetic_dist))
    test_app = FastAPI()
    resolved = attach_dashboard(test_app)
    assert resolved == synthetic_dist

    client = TestClient(test_app)
    r = client.get(f"{DASHBOARD_URL_PATH}/")
    assert r.status_code == 200
    assert "pcil-dashboard-fixture" in r.text


def test_root_metadata_advertises_dashboard(client) -> None:
    """The service metadata always advertises the dashboard URL.

    Whether the assets are actually mounted is reported separately via
    `dashboard_available`; the URL itself stays stable so callers can
    link to it from external docs.
    """
    body = client.get("/").json()
    assert body["endpoints"]["dashboard"] == f"GET {DASHBOARD_URL_PATH}/"

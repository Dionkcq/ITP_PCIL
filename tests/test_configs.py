"""Tests for the config recipe endpoints (dashboard config editor).

GET  /configs           — list recipes under machines/
GET  /configs/load      — parse a recipe into structured data
POST /configs/validate  — dry-run validation, nothing written
POST /configs/save      — validate + persist (backup on overwrite)

All tests run against a copy of the real inkjet config in a tmp dir
(`machines_root` fixture monkeypatches MACHINES_ROOT), so they never
touch the repo's machines/ folder.
"""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def machines_root(monkeypatch, tmp_path) -> Path:
    """A tmp machines/ folder seeded with the real inkjet config."""
    import pcil.orchestrator as orch

    root = tmp_path / "machines"
    machine_dir = root / "inkjet_printer"
    machine_dir.mkdir(parents=True)

    real = Path(__file__).resolve().parents[1] / "machines" / "inkjet_printer" / "config.yaml"
    (machine_dir / "config.yaml").write_text(
        real.read_text(encoding="utf-8"), encoding="utf-8"
    )

    monkeypatch.setattr(orch, "MACHINES_ROOT", root)
    return root


def _load_payload(client, recipe="inkjet_printer/config.yaml"):
    """Fetch a recipe through the API and return its config dict."""
    r = client.get("/configs/load", params={"path": recipe})
    assert r.status_code == 200
    return r.json()["config"]


# ── listing + loading ────────────────────────────────────────


def test_list_configs(client, machines_root):
    r = client.get("/configs")
    assert r.status_code == 200
    configs = r.json()["configs"]
    assert len(configs) == 1
    assert configs[0]["machine"] == "inkjet_printer"
    assert configs[0]["recipe"] == "inkjet_printer/config.yaml"
    # The path string usable directly as /pipeline/run's config_path.
    assert configs[0]["config_path"] == "machines/inkjet_printer/config.yaml"


def test_load_returns_structured_config(client, machines_root):
    cfg = _load_payload(client)
    assert cfg["input"]["timestamp_column"] == "timestamp"
    assert "oee" in cfg["input"]["targets"]
    assert "air_pressure_low_ratio" in cfg["input"]["numerical_features"]
    assert isinstance(cfg["feature_descriptions"], dict)


def test_load_accepts_machines_prefixed_path(client, machines_root):
    """The /pipeline/run-style path spelling works too."""
    r = client.get(
        "/configs/load", params={"path": "machines/inkjet_printer/config.yaml"}
    )
    assert r.status_code == 200


def test_load_rejects_traversal_and_missing(client, machines_root):
    r = client.get("/configs/load", params={"path": "../pcil/orchestrator.yaml"})
    assert r.status_code == 400

    r = client.get("/configs/load", params={"path": "inkjet_printer/config.txt"})
    assert r.status_code == 400  # non-YAML suffix

    r = client.get("/configs/load", params={"path": "inkjet_printer/nope.yaml"})
    assert r.status_code == 404


# ── validation ───────────────────────────────────────────────


def test_save_invalid_payload_writes_nothing(client, machines_root):
    """Garbage in -> status 'invalid' + error list, file untouched."""
    target = machines_root / "inkjet_printer" / "config.yaml"
    before = target.read_text(encoding="utf-8")

    cfg = _load_payload(client)
    cfg["input"]["targets"] = []                       # no targets
    cfg["trigger"]["mode"] = "sometimes"               # bogus mode
    cfg["input"]["numerical_features"].append("oee")   # duplicate vs targets... wait, targets now empty
    cfg["input"]["numerical_features"].append("vibration")  # duplicate feature

    r = client.post(
        "/configs/save",
        json={"path": "inkjet_printer/config.yaml", "config": cfg},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "invalid"
    joined = " ".join(body["errors"])
    assert "targets" in joined
    assert "trigger.mode" in joined
    assert "duplicate" in joined
    assert target.read_text(encoding="utf-8") == before


def test_validate_endpoint_is_dry_run(client, machines_root):
    target = machines_root / "inkjet_printer" / "config.yaml"
    before = target.read_text(encoding="utf-8")

    cfg = _load_payload(client)
    cfg["trigger"]["mode"] = "last_n"
    cfg["trigger"]["last_n"] = -5

    r = client.post(
        "/configs/validate",
        json={"path": "inkjet_printer/config.yaml", "config": cfg},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "invalid"
    assert any("last_n" in e for e in body["errors"])
    assert target.read_text(encoding="utf-8") == before


def test_time_range_mode_requires_valid_bounds(client, machines_root):
    cfg = _load_payload(client)
    cfg["trigger"]["mode"] = "time_range"
    cfg["trigger"]["start_time"] = "not-a-date"
    cfg["trigger"]["end_time"] = "2026-06-15T10:00:00+00:00"

    r = client.post(
        "/configs/validate",
        json={"path": "inkjet_printer/config.yaml", "config": cfg},
    )
    body = r.json()
    assert body["status"] == "invalid"
    assert any("start_time" in e for e in body["errors"])


# ── saving ───────────────────────────────────────────────────


def test_save_add_sensor_roundtrip_with_backup(client, machines_root):
    """The headline use case: add a sensor via the editor.

    Verifies the new feature + description round-trip, a timestamped
    backup of the previous version is stored, and a hand-added unknown
    top-level key survives the save.
    """
    target = machines_root / "inkjet_printer" / "config.yaml"
    # Simulate an engineer's hand-added custom key.
    target.write_text(
        target.read_text(encoding="utf-8") + "\ncustom_note: keep-me\n",
        encoding="utf-8",
    )

    cfg = _load_payload(client)
    cfg["input"]["numerical_features"].append("new_sensor")
    cfg["feature_descriptions"]["new_sensor"] = "Reading from the newly added sensor."

    r = client.post(
        "/configs/save",
        json={"path": "inkjet_printer/config.yaml", "config": cfg},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backup"] and ".backups/" in body["backup"]

    # Round-trip: the saved file parses and contains the new sensor.
    saved = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert "new_sensor" in saved["input"]["numerical_features"]
    assert saved["feature_descriptions"]["new_sensor"].startswith("Reading")
    assert saved["custom_note"] == "keep-me"

    backups = list((machines_root / "inkjet_printer" / ".backups").glob("*.yaml"))
    assert len(backups) == 1
    # The backup is the PRE-save version: no new_sensor in it.
    old = yaml.safe_load(backups[0].read_text(encoding="utf-8"))
    assert "new_sensor" not in old["input"]["numerical_features"]


def test_save_as_creates_new_recipe(client, machines_root):
    target = machines_root / "inkjet_printer" / "config.yaml"
    before = target.read_text(encoding="utf-8")

    cfg = _load_payload(client)
    cfg["trigger"]["mode"] = "last_n"
    cfg["trigger"]["last_n"] = 120

    r = client.post(
        "/configs/save",
        json={
            "path": "inkjet_printer/config.yaml",
            "config": cfg,
            "save_as": "config_last120",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["recipe"] == "inkjet_printer/config_last120.yaml"

    # Original untouched; new recipe exists, parses, and is listed.
    assert target.read_text(encoding="utf-8") == before
    new_file = machines_root / "inkjet_printer" / "config_last120.yaml"
    assert yaml.safe_load(new_file.read_text(encoding="utf-8"))["trigger"]["last_n"] == 120
    names = [c["name"] for c in client.get("/configs").json()["configs"]]
    assert "config_last120.yaml" in names


def test_save_as_rejects_bad_names(client, machines_root):
    cfg = _load_payload(client)
    for bad in ("../evil", "a/b", "name with spaces", ""):
        r = client.post(
            "/configs/save",
            json={
                "path": "inkjet_printer/config.yaml",
                "config": cfg,
                "save_as": bad,
            },
        )
        # "" -> falsy save_as is treated as a plain overwrite, so only
        # the genuinely malformed names must 400.
        if bad:
            assert r.status_code == 400, f"save_as={bad!r} should be rejected"

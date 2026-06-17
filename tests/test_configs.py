"""Tests for the config recipe endpoints (dashboard config editor).

GET  /configs           — list recipes under systems/
GET  /configs/load      — parse a recipe into structured data
POST /configs/validate  — dry-run validation, nothing written
POST /configs/save      — validate + persist (backup on overwrite)

All tests run against a copy of the real inkjet config in a tmp dir
(`machines_root` fixture monkeypatches SYSTEMS_ROOT), so they never
touch the repo's systems/ folder.
"""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def machines_root(monkeypatch, tmp_path) -> Path:
    """A tmp systems/ folder seeded with the real inkjet config."""
    import pcil.orchestrator as orch

    root = tmp_path / "systems"
    machine_dir = root / "inkjet_printer"
    machine_dir.mkdir(parents=True)

    real = Path(__file__).resolve().parents[1] / "systems" / "inkjet_printer" / "config.yaml"
    (machine_dir / "config.yaml").write_text(
        real.read_text(encoding="utf-8"), encoding="utf-8"
    )

    monkeypatch.setattr(orch, "SYSTEMS_ROOT", root)
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
    assert configs[0]["config_path"] == "systems/inkjet_printer/config.yaml"


def test_load_returns_structured_config(client, machines_root):
    cfg = _load_payload(client)
    assert cfg["input"]["timestamp_column"] == "timestamp"
    assert "oee" in cfg["input"]["targets"]
    assert "air_pressure_low_ratio" in cfg["input"]["numerical_features"]
    assert isinstance(cfg["feature_descriptions"], dict)


def test_load_accepts_machines_prefixed_path(client, machines_root):
    """The /pipeline/run-style path spelling works too."""
    r = client.get(
        "/configs/load", params={"path": "systems/inkjet_printer/config.yaml"}
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


# ── creating a new machine ───────────────────────────────────


def test_create_new_machine(client, machines_root):
    """POST /configs/create makes systems/<machine>/<name>.yaml and the
    new recipe shows up in GET /configs."""
    cfg = _load_payload(client)
    cfg["system"] = "laser_welder"

    r = client.post(
        "/configs/create",
        json={"machine": "laser_welder", "name": "config", "config": cfg},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["recipe"] == "laser_welder/config.yaml"
    assert body["config_path"] == "systems/laser_welder/config.yaml"

    new_file = machines_root / "laser_welder" / "config.yaml"
    saved = yaml.safe_load(new_file.read_text(encoding="utf-8"))
    assert saved["system"] == "laser_welder"

    recipes = [c["recipe"] for c in client.get("/configs").json()["configs"]]
    assert "laser_welder/config.yaml" in recipes


def test_create_refuses_overwrite(client, machines_root):
    cfg = _load_payload(client)
    r = client.post(
        "/configs/create",
        json={"machine": "inkjet_printer", "name": "config", "config": cfg},
    )
    assert r.status_code == 409
    assert "already exists" in r.json()["detail"]


def test_create_rejects_bad_machine_names(client, machines_root):
    cfg = _load_payload(client)
    for bad in ("../evil", "a/b", "name with spaces", ".hidden"):
        r = client.post(
            "/configs/create",
            json={"machine": bad, "config": cfg},
        )
        assert r.status_code == 400, f"machine={bad!r} should be rejected"
    assert not (machines_root / "..").joinpath("evil").exists()


def test_create_invalid_config_creates_nothing(client, machines_root):
    cfg = _load_payload(client)
    cfg["input"]["targets"] = []
    r = client.post(
        "/configs/create",
        json={"machine": "broken_machine", "config": cfg},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "invalid"
    assert not (machines_root / "broken_machine").exists()


# ── deleting recipes ─────────────────────────────────────────


def test_delete_recipe_is_recoverable(client, machines_root):
    """Delete moves the file into .backups/ (recoverable), and the
    recipe disappears from GET /configs."""
    r = client.post(
        "/configs/delete", json={"path": "inkjet_printer/config.yaml"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["deleted"] == "inkjet_printer/config.yaml"
    assert ".backups/" in body["backup"] and ".deleted-" in body["backup"]

    assert not (machines_root / "inkjet_printer" / "config.yaml").exists()
    backups = list(
        (machines_root / "inkjet_printer" / ".backups").glob("config.deleted-*.yaml")
    )
    assert len(backups) == 1
    # The moved file is intact YAML — restoring = moving it back.
    assert yaml.safe_load(backups[0].read_text(encoding="utf-8"))["input"]

    assert client.get("/configs").json()["configs"] == []


def test_delete_missing_and_traversal(client, machines_root):
    r = client.post("/configs/delete", json={"path": "inkjet_printer/nope.yaml"})
    assert r.status_code == 404
    r = client.post("/configs/delete", json={"path": "../pcil/orchestrator.yaml"})
    assert r.status_code == 400


# ── anomaly bundle listing ───────────────────────────────────


def test_list_anomaly_models(client, isolated_data_dir):
    """GET /anomaly/models parses <model_type>_<model_id>.pkl filenames,
    including the underscore-bearing non_cyclical prefix, and ignores
    anything that isn't a recognised bundle."""
    for name in (
        "cyclical_inkjet_01.pkl",
        "non_cyclical_inkjet_01.pkl",
        "irregular_conveyor_a_b.pkl",
        "random.pkl",          # no recognised prefix
        "notes.txt",           # not a .pkl
    ):
        (isolated_data_dir / name).write_bytes(b"x")

    r = client.get("/anomaly/models")
    assert r.status_code == 200
    models = {(m["model_type"], m["model_id"]) for m in r.json()["models"]}
    assert models == {
        ("cyclical", "inkjet_01"),
        ("non_cyclical", "inkjet_01"),
        ("irregular", "conveyor_a_b"),
    }
    for m in r.json()["models"]:
        assert m["file"].endswith(".pkl")
        assert "size_kb" in m and "modified" in m

"""Tests for POST /anomaly/train.

The endpoint supports two combinations:

  cyclical    + normal_only       -> uses Jaymon's IsolationForest
  non_cyclical + clean_vs_anomaly  -> uses Zi Hin's RandomForest

The cyclical real-training path needs ~200 rows per cycle to actually
detect cycles, which is more than we want to commit as a fixture, so
those tests monkeypatch the training function. The non-cyclical real
path works fine on the 50-row fixtures with a smaller window_size_rows.
"""


# ─────────────────────────────────────────────────────────────────
# Validation paths — these don't need training to run
# ─────────────────────────────────────────────────────────────────

def test_train_unsupported_combination_returns_400(client):
    """Unsupported (model_type, training_mode) pair should fail fast
    with a clear message listing what IS supported."""
    r = client.post(
        "/anomaly/train",
        data={
            "model_type": "cyclical",
            "training_mode": "clean_vs_anomaly",  # only valid for non_cyclical
            "model_id": "anything",
        },
    )
    assert r.status_code == 400
    assert "unsupported combination" in r.json()["detail"]


def test_train_cyclical_without_file_returns_400(client):
    r = client.post(
        "/anomaly/train",
        data={
            "model_type": "cyclical",
            "training_mode": "normal_only",
            "model_id": "anything",
        },
    )
    assert r.status_code == 400
    assert "file" in r.json()["detail"].lower()


def test_train_cyclical_rejects_missing_required_columns(
    client, isolated_data_dir,
):
    """An upload that doesn't have the machine_id / signal_value /
    timestamp columns should fail with 400 listing what's missing."""
    bad_csv = b"foo,bar\n1,2\n3,4\n"  # nothing the model can use
    r = client.post(
        "/anomaly/train",
        data={
            "model_type": "cyclical",
            "training_mode": "normal_only",
            "model_id": "test_validation",
        },
        files={"file": ("bad.csv", bad_csv, "text/csv")},
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "missing required columns" in detail
    # The missing columns should all be named
    for expected in ("machine_id", "signal_value", "timestamp"):
        assert expected in detail


def test_train_non_cyclical_without_files_returns_400(client):
    r = client.post(
        "/anomaly/train",
        data={
            "model_type": "non_cyclical",
            "training_mode": "clean_vs_anomaly",
            "model_id": "anything",
        },
    )
    assert r.status_code == 400
    assert "clean_file" in r.json()["detail"] or "anomaly_file" in r.json()["detail"]


# ─────────────────────────────────────────────────────────────────
# Happy-path: cyclical with a monkeypatched train()
# ─────────────────────────────────────────────────────────────────

def test_train_cyclical_saves_bundle_via_monkeypatch(
    monkeypatch, client, cyclical_tiny_path, isolated_data_dir,
):
    """Verify the cyclical training endpoint:
      - accepts the upload,
      - validates columns,
      - calls cyclical.train.train() with the parsed DataFrame,
      - dumps the returned bundle to data/cyclical_<model_id>.pkl,
      - returns a friendly response.

    The real cyclical train() requires ~200+ row cycles which our 50-row
    fixture can't produce, so we replace it with a stub that returns a
    minimal valid bundle.
    """
    fake_bundle_marker = {"model": "FAKE_CYCLICAL_BUNDLE", "feature_columns": ["a"]}

    def fake_train(df, model_name, *, machine_id_column, signal_column,
                    timestamp_column, model_kwargs=None):
        assert len(df) == 50          # confirm the fixture made it through
        assert model_name == "isolation_forest"
        assert machine_id_column == "machine_id"
        assert signal_column == "signal_value"
        assert timestamp_column == "timestamp"
        return fake_bundle_marker

    import pcil.utils.anomaly.cyclical.train as ct_module
    monkeypatch.setattr(ct_module, "train", fake_train)

    with open(cyclical_tiny_path, "rb") as f:
        r = client.post(
            "/anomaly/train",
            data={
                "model_type": "cyclical",
                "training_mode": "normal_only",
                "model_id": "test_cyclical",
                "model_name": "isolation_forest",
                "machine_id_column": "machine_id",
                "signal_column": "signal_value",
                "timestamp_column": "timestamp",
            },
            files={"file": ("cyclical_tiny.csv", f, "text/csv")},
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_type"] == "cyclical"
    assert body["model_id"] == "test_cyclical"
    assert body["input_rows"] == 50

    # The bundle should be on disk where /anomaly/score expects to find it.
    bundle_path = isolated_data_dir / "cyclical_test_cyclical.pkl"
    assert bundle_path.is_file()

    # And the bundle contents should be what fake_train returned.
    import joblib
    loaded = joblib.load(bundle_path)
    assert loaded == fake_bundle_marker


# ─────────────────────────────────────────────────────────────────
# Happy-path: non-cyclical with real training on tiny fixtures
# ─────────────────────────────────────────────────────────────────

def test_train_non_cyclical_clean_vs_anomaly_saves_real_bundle(
    client, non_cyclical_clean_tiny_path, non_cyclical_anomaly_tiny_path,
    isolated_data_dir,
):
    """Real training path: 50-row clean + 50-row anomaly fixtures,
    window_size_rows=20 -> 2 windows per recording = 4 training windows.

    This proves the refactored train_from_clean_and_anomaly works
    end-to-end through the API. The resulting bundle is loadable and
    has the keys score.py / /anomaly/score expect."""
    with open(non_cyclical_clean_tiny_path, "rb") as cf, \
         open(non_cyclical_anomaly_tiny_path, "rb") as af:
        r = client.post(
            "/anomaly/train",
            data={
                "model_type": "non_cyclical",
                "training_mode": "clean_vs_anomaly",
                "model_id": "test_noncyc",
                "window_size_rows": "20",   # small enough for 50-row fixtures
                "train_ratio": "1.0",       # use all data for training (tiny set)
            },
            files={
                "clean_file":   ("clean.csv",   cf, "text/csv"),
                "anomaly_file": ("anomaly.csv", af, "text/csv"),
            },
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_type"] == "non_cyclical"
    assert body["input_rows"] == 100  # 50 clean + 50 anomaly

    # Bundle should be on disk + have the expected keys for score.py.
    bundle_path = isolated_data_dir / "non_cyclical_test_noncyc.pkl"
    assert bundle_path.is_file()
    import joblib
    bundle = joblib.load(bundle_path)
    for k in ("model", "normaliser", "feature_columns",
               "machine_id_column", "window_size_rows", "channel_columns"):
        assert k in bundle, f"bundle missing key: {k}"


def test_train_non_cyclical_handles_too_short_recording(
    client, non_cyclical_clean_tiny_path, non_cyclical_anomaly_tiny_path,
    isolated_data_dir,
):
    """If the recording is too short for the chosen window_size_rows
    to produce any windows, the API should fail with a clear 400."""
    with open(non_cyclical_clean_tiny_path, "rb") as cf, \
         open(non_cyclical_anomaly_tiny_path, "rb") as af:
        r = client.post(
            "/anomaly/train",
            data={
                "model_type": "non_cyclical",
                "training_mode": "clean_vs_anomaly",
                "model_id": "test_noncyc_short",
                "window_size_rows": "10000",  # way bigger than 50 rows
                "train_ratio": "0.8",
            },
            files={
                "clean_file":   ("clean.csv",   cf, "text/csv"),
                "anomaly_file": ("anomaly.csv", af, "text/csv"),
            },
        )

    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "no windows" in detail.lower() or "window_size_rows" in detail

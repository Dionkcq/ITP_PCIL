"""Tests for the irregular-sampling anomaly pipeline.

Unit level: synthetic event streams (jittered Poisson arrivals) train a
bundle, then an eval stream containing a BURST (event flood) and a
STALL (reporting outage) must score those windows above the normal
baseline. This demonstrates the pipeline end-to-end the same way the
cyclical/non-cyclical pipelines were demoed on their datasets — there
is no real irregular dataset in the project yet (the pipeline
definition is the deliverable, per Winardi's pipeline-vs-instance
framing).

Endpoint level: /anomaly/train and /anomaly/score with
model_type=irregular, using isolated_data_dir so bundles never touch
the real ITP/data/ folder.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from pcil.utils.anomaly.irregular.score import score as irregular_score
from pcil.utils.anomaly.irregular.train import train as irregular_train


def _make_events(
    start: str,
    duration_s: float,
    rate_hz: float,
    *,
    machine: str = "inkjet_01",
    seed: int = 0,
    value_mean: float = 5.0,
) -> pd.DataFrame:
    """Jittered-Poisson event stream: irregular gaps around 1/rate_hz."""
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / rate_hz, size=int(duration_s * rate_hz * 2))
    times = np.cumsum(gaps)
    times = times[times < duration_s]
    return pd.DataFrame({
        "machine_id": machine,
        "timestamp": pd.Timestamp(start) + pd.to_timedelta(times, unit="s"),
        "signal_value": rng.normal(value_mean, 0.5, size=len(times)),
    })


def _train_bundle(seed: int = 0) -> dict:
    train_df = _make_events("2026-06-01 08:00:00", 60.0, 10.0, seed=seed)
    return irregular_train(
        train_df,
        window_seconds=1.0,
        value_column="signal_value",
    )


def test_train_returns_complete_bundle():
    bundle = _train_bundle()
    assert bundle["model_name"] == "isolation_forest"
    assert bundle["window_seconds"] == 1.0
    assert bundle["value_column"] == "signal_value"
    assert bundle["trained_machine_ids"] == ["inkjet_01"]
    assert "event_count" in bundle["feature_columns"]
    assert "mean_interval" in bundle["feature_columns"]
    assert "value_mean" in bundle["feature_columns"]
    assert isinstance(bundle["threshold"], float)


def test_burst_and_stall_score_above_normal_windows():
    bundle = _train_bundle()

    # Eval stream: 10 s normal, then a 1 s burst (~15x event rate),
    # then a 5 s stall (zero events), then 4 s normal again.
    normal_a = _make_events("2026-06-02 08:00:00", 10.0, 10.0, seed=1)
    burst = _make_events("2026-06-02 08:00:10", 1.0, 150.0, seed=2)
    normal_b = _make_events("2026-06-02 08:00:16", 4.0, 10.0, seed=3)
    eval_df = pd.concat([normal_a, burst, normal_b], ignore_index=True)

    scored = irregular_score(eval_df, bundle)

    burst_windows = scored[scored["event_count"] > 50]
    stall_windows = scored[scored["event_count"] == 0]
    normal_windows = scored[
        (scored["event_count"] >= 5) & (scored["event_count"] <= 20)
    ]
    assert len(burst_windows) >= 1
    assert len(stall_windows) >= 4  # the 5 s gap between burst and normal_b
    assert len(normal_windows) >= 10

    # Out-of-distribution windows must clearly outscore the normal
    # baseline. (Not "outscore every normal window": IsolationForest
    # scores saturate for far outliers, so a single odd-but-normal
    # window can land within a few percent of a true anomaly.)
    normal_median = normal_windows["anomaly_score"].median()
    assert burst_windows["anomaly_score"].min() > normal_median
    assert stall_windows["anomaly_score"].min() > normal_median

    # The stored 95th-percentile threshold should flag them...
    assert burst_windows["is_anomaly"].all()
    assert stall_windows["is_anomaly"].all()

    # ...while the majority of normal windows are NOT flagged.
    assert normal_windows["is_anomaly"].mean() < 0.5


def test_score_keeps_window_timestamps():
    bundle = _train_bundle()
    eval_df = _make_events("2026-06-02 09:00:00", 5.0, 10.0, seed=4)
    scored = irregular_score(eval_df, bundle)
    assert "window_start_timestamp" in scored.columns
    # Windows anchor on the first event seen for the machine.
    assert scored["window_start_timestamp"].iloc[0] == eval_df["timestamp"].min()


# ─────────────────────────────────────────────────────────────
# Endpoint tests
# ─────────────────────────────────────────────────────────────

def _train_via_endpoint(client, model_id: str = "test01") -> object:
    train_df = _make_events("2026-06-01 08:00:00", 30.0, 10.0, seed=5)
    csv_bytes = train_df.to_csv(index=False).encode()
    return client.post(
        "/anomaly/train",
        data={
            "model_type": "irregular",
            "training_mode": "normal_only",
            "model_id": model_id,
            "window_seconds": "1.0",
            "value_column": "signal_value",
        },
        files={"file": ("irregular_train.csv", io.BytesIO(csv_bytes), "text/csv")},
    )


def test_train_endpoint_saves_irregular_bundle(client, isolated_data_dir):
    r = _train_via_endpoint(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_type"] == "irregular"
    assert (isolated_data_dir / "irregular_test01.pkl").is_file()


def test_score_endpoint_returns_flags_and_threshold(client, isolated_data_dir):
    assert _train_via_endpoint(client).status_code == 200

    eval_df = _make_events("2026-06-02 08:00:00", 5.0, 10.0, seed=6)
    eval_df["timestamp"] = eval_df["timestamp"].astype(str)  # JSON-safe
    r = client.post(
        "/anomaly/score",
        json={
            "data": eval_df.to_dict(orient="records"),
            "model_type": "irregular",
            "model_id": "test01",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_type"] == "irregular"
    assert body["windows_scored"] >= 5
    assert len(body["anomaly_scores"]) == body["windows_scored"]
    assert len(body["is_anomaly"]) == body["windows_scored"]
    assert isinstance(body["threshold"], float)
    assert len(body["window_start_timestamps"]) == body["windows_scored"]


def test_score_endpoint_404_when_bundle_missing(client, isolated_data_dir):
    r = client.post(
        "/anomaly/score",
        json={
            "data": [{"timestamp": "2026-06-02T08:00:00", "machine_id": "x"}],
            "model_type": "irregular",
            "model_id": "definitely_not_trained",
        },
    )
    assert r.status_code == 404
    assert "irregular bundle not found" in r.json()["detail"]


def test_score_endpoint_400_when_columns_missing(client, isolated_data_dir):
    assert _train_via_endpoint(client).status_code == 200

    r = client.post(
        "/anomaly/score",
        json={
            "data": [{"wrong_column": 1.0}],
            "model_type": "irregular",
            "model_id": "test01",
        },
    )
    assert r.status_code == 400
    assert "missing required columns" in r.json()["detail"]

"""
Fixture generator — run once to produce the committed test CSVs.
=================================================================
The CSVs in this folder are checked into git. This script exists so
they can be reproduced or extended (e.g. add a new column to the
shop-floor fixture) without writing the rows by hand.

Run once from PCIL_dev/:
    python tests/fixtures/_generate.py

Generated:
    shop_floor_tiny.csv          — matches machines/inkjet_printer/config.yaml schema
    cyclical_tiny.csv            — small cyclical signal for cyclical training tests
    non_cyclical_clean_tiny.csv  — clean acoustic-style recording
    non_cyclical_anomaly_tiny.csv — anomaly acoustic-style recording
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def write_shop_floor_tiny() -> None:
    """50 rows matching the inkjet_printer config.yaml schema. Enough
    variation for LinearRegression to fit without a singular matrix.
    """
    n = 50
    rng = np.random.default_rng(seed=42)
    df = pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-06-12T09:00:00+00:00", periods=n, freq="1s"
        ),
        "availability": np.clip(1.0 - rng.uniform(0, 0.2, n), 0, 1),
        "performance":  np.clip(1.0 - rng.uniform(0, 0.3, n), 0, 1),
        "quality":      np.clip(0.95 + rng.uniform(0, 0.05, n), 0, 1),
        "oee":          np.clip(rng.uniform(0.5, 1.0, n), 0, 1),
        "air_pressure_low_ratio":   rng.uniform(0.0, 0.3, n).round(4),
        "cycle_stop_present":       rng.binomial(1, 0.1, n),
        "vibration":                rng.uniform(0.0, 0.1, n).round(5),
        "acoustic_emission":        rng.uniform(0.1, 0.3, n).round(4),
        "setpressure_cycle_count":  rng.integers(0, 250, n),
        "setvelo_mean":             rng.uniform(-15, 10, n).round(3),
        "cyclical_anomaly_score":   rng.uniform(0.0, 0.2, n).round(4),
        "acoustic_anomaly_score":   rng.uniform(0.0, 0.5, n).round(4),
    })
    out = HERE / "shop_floor_tiny.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} — {len(df)} rows")


def write_cyclical_tiny() -> None:
    """50 rows of a fake cyclical pressure signal. Two machines, sine
    wave + small noise. Not enough rows to detect cycles under the
    default min_cycle_samples=200 — that's by design: most cyclical
    tests monkeypatch the heavy training step, and rely on this CSV
    only to verify upload + column-validation plumbing.
    """
    rng = np.random.default_rng(seed=7)
    n_per_machine = 25
    rows = []
    for machine_id in ("inkjet_01", "inkjet_02"):
        t = np.arange(n_per_machine)
        signal = 0.5 + 0.4 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.02, n_per_machine)
        for i in range(n_per_machine):
            rows.append({
                "machine_id": machine_id,
                "timestamp":  f"2026-06-12T09:00:{i:02d}+00:00",
                "signal_value": round(float(signal[i]), 4),
            })
    df = pd.DataFrame(rows)
    out = HERE / "cyclical_tiny.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} — {len(df)} rows")


def _make_acoustic(n: int, rms_target: float, seed: int) -> pd.DataFrame:
    """Acoustic-style recording with the 4 channel columns Zi Hin's
    features.py expects. rms_target sets the rough amplitude so clean
    vs anomaly are distinguishable.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Sample": np.arange(n),
        "Time (s)": (np.arange(n) / 25600.0).round(8),
        "Acceleration 0 (g)": rng.normal(0, rms_target, n).round(5),
        "Acceleration 1 (g)": rng.normal(0, rms_target, n).round(5),
        "Acceleration 2 (g)": rng.normal(0, rms_target, n).round(5),
        "AE (V) (V)":         (0.2 + rng.normal(0, rms_target, n)).round(5),
    })


def write_non_cyclical_tinys() -> None:
    """Two 50-row acoustic-style CSVs. Clean has low RMS; anomaly has
    high RMS. Tests will pass window_size_rows=20 so each file produces
    ~2 windows — enough for the supervised training step to fit.
    """
    clean = _make_acoustic(n=50, rms_target=0.02, seed=11)
    anomaly = _make_acoustic(n=50, rms_target=0.20, seed=12)

    out_clean = HERE / "non_cyclical_clean_tiny.csv"
    out_anomaly = HERE / "non_cyclical_anomaly_tiny.csv"
    clean.to_csv(out_clean, index=False)
    anomaly.to_csv(out_anomaly, index=False)
    print(f"wrote {out_clean} — {len(clean)} rows")
    print(f"wrote {out_anomaly} — {len(anomaly)} rows")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    write_shop_floor_tiny()
    write_cyclical_tiny()
    write_non_cyclical_tinys()


if __name__ == "__main__":
    main()

"""
build_mock_shop_floor.py
========================

Generate `data/mock_shop_floor.csv` — a synthetic shop-floor DataFrame that
matches the schema Pipeline #1's `preprocess.py` expects (see
`machines/inkjet_printer/config.yaml`).

Why this exists
---------------
The real shop-floor DB will be owned by the engineering team and populated
from many live machines + the cyclical/non-cyclical anomaly pipelines.
Until that exists, we need a realistic-shaped input to develop and demo
the runtime path (trigger -> preprocess -> adapter -> score -> RAG -> LLM)
without waiting for engineering's Postgres to come online.

Inputs (real sensor data, gitignored under `data/`)
---------------------------------------------------
1. `Clean_Data.csv` — cyclical pressure, 1 ms resolution (~2 min)
2. `Inkjet Printer Data Collection/Acoustic Sensor Data/machine_on_clean.csv`
   and `machine_on_anomaly.csv` — 25.6 kHz accelerometer + AE
3. `Inkjet Printer Data Collection/Air Pressure Low.csv` — 1 ms 0/1 flags

Cadence mismatch
----------------
Native cadences differ (1 ms / 25.6 kHz / 1 ms) and timestamps don't overlap
across sources. This script:
  - Aggregates each source independently up to per-second summary rows.
  - Re-stamps everything onto one synthetic 120-second grid starting at
    `ORIGIN_TIMESTAMP` so the result is a coherent slice.

Per-second is an assumed canonical row cadence. Pending Winardi confirmation
(Q11 in `notes/CONTEXT_HANDOFF.md`).

Two machines
------------
`inkjet_02` is fabricated from `inkjet_01` with small per-machine offsets so
the per-machine normaliser in the anomaly pipelines has something to do.

Targets (`availability`, `performance`, `quality`, `oee`)
---------------------------------------------------------
Faked via a simple heuristic — when error flags fire, OEE drops. The real
shop-floor DB will have these computed upstream. Documented in the schema.

Anomaly score columns (placeholders)
------------------------------------
`cyclical_anomaly_score` and `acoustic_anomaly_score` would normally be
produced by the team's cyclical / non-cyclical anomaly pipelines and
written into the shop-floor DB upstream of our trigger. Those pipelines
aren't built yet, so the mock fakes them:
  - `acoustic_anomaly_score`: ground-truth-driven. Low (~0.1) for rows
    sourced from `machine_on_clean.csv`, high (~0.7) for rows sourced
    from `machine_on_anomaly.csv`, plus small noise.
  - `cyclical_anomaly_score`: heuristic. Small baseline scaled from
    per-second SetPressure variability + a programmed bump in seconds
    80-95 to simulate the cyclical pipeline catching a different event
    from the acoustic one. Gives the trigger two distinct events.
Both placeholders get replaced by real `score()` output once the
anomaly pipelines train.

Output
------
`data/mock_shop_floor.csv` — columns:
    timestamp, machine_id,
    availability, performance, quality, oee,
    air_pressure_low_ratio, cycle_stop_present,
    vibration, acoustic_emission,
    setpressure_cycle_count, setvelo_mean,
    cyclical_anomaly_score, acoustic_anomaly_score

This is a one-off mock. Discard when the engineering team's real shop-floor
DB exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "mock_shop_floor.csv"

ORIGIN_TIMESTAMP = pd.Timestamp("2026-05-15 09:00:00+00:00")
DURATION_SECONDS = 120


def rms(x: pd.Series) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0


def aggregate_cyclical(path: Path, n_seconds: int) -> pd.DataFrame:
    """1 ms cyclical pressure -> per-second mean velocity + pressure-change count
    + placeholder cyclical anomaly score.

    `setpressure_cycle_count` is a proxy: number of times SetPressure changes by
    more than 0.01 in the second. The raw data has 4-row repetition (1 kHz
    logger oversampling a ~250 Hz source), so a value-change roughly maps to
    one cycle of the underlying signal.

    `cyclical_anomaly_score` is a placeholder: per-second SetPressure std
    rescaled to a small baseline, plus a programmed bump in seconds 80-95
    so the trigger has a cyclical event distinct from the acoustic one.
    """
    df = pd.read_csv(path, sep=";")
    df["_time"] = pd.to_datetime(df["_time"], format="ISO8601")
    df = df.set_index("_time")

    out = pd.DataFrame()
    out["setvelo_mean"] = df["SetVelo"].resample("1s").mean()
    out["setpressure_cycle_count"] = (
        df["SetPressure"]
        .resample("1s")
        .apply(lambda s: int(s.diff().abs().gt(0.01).sum()))
    )

    # Placeholder anomaly score: baseline from SetPressure variability...
    pressure_std = df["SetPressure"].resample("1s").std().fillna(0)
    baseline = (pressure_std / (pressure_std.max() or 1.0)) * 0.15

    # ...plus a programmed bump in seconds 80-95 to simulate a cyclical event.
    second_idx = np.arange(len(baseline))
    bump = np.where((second_idx >= 80) & (second_idx < 95), 0.6, 0.0)
    out["cyclical_anomaly_score"] = (baseline.values + bump).clip(0, 1)

    return out.head(n_seconds).fillna(0).reset_index(drop=True)


def _load_acoustic(path: Path, second_offset: int, max_seconds: int, is_anomaly: bool) -> pd.DataFrame:
    """Read one acoustic CSV, group 25.6 kHz samples into per-second RMS rows.

    Stamps each row with an `is_anomaly` ground-truth label so the caller can
    drive the placeholder `acoustic_anomaly_score`.
    """
    df = pd.read_csv(path, skiprows=5)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df["second"] = df["Time (s)"].astype(int) + second_offset
    grouped = df.groupby("second").agg(
        vibration=("Acceleration 0 (g)", rms),
        acoustic_emission=("AE (V) (V)", rms),
    )
    grouped = grouped[grouped.index < second_offset + max_seconds].reset_index(drop=True)
    grouped["_is_anomaly"] = int(is_anomaly)
    return grouped


def aggregate_acoustic(
    clean_path: Path, anomaly_path: Path, n_seconds: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Clean first half, anomaly second half -> per-second RMS vibration + AE
    + placeholder acoustic anomaly score.

    Concatenates the two files so the trigger has a visible clean->anomaly
    transition to fire on. The placeholder `acoustic_anomaly_score` is
    ground-truth-driven: ~0.1 for clean rows, ~0.7 for anomaly rows, with
    small noise. This is what the non-cyclical pipeline would produce.
    """
    half = n_seconds // 2
    clean = _load_acoustic(clean_path, second_offset=0, max_seconds=half, is_anomaly=False)
    anomaly = _load_acoustic(
        anomaly_path, second_offset=half, max_seconds=n_seconds - half, is_anomaly=True
    )
    out = pd.concat([clean, anomaly], ignore_index=True)

    noise = rng.normal(0, 0.04, size=len(out))
    base = np.where(out["_is_anomaly"] == 1, 0.70, 0.10)
    out["acoustic_anomaly_score"] = np.clip(base + noise, 0, 1)
    out = out.drop(columns="_is_anomaly")
    return out


def aggregate_flags(path: Path, n_seconds: int) -> pd.DataFrame:
    """1 ms 0/1 error flags -> per-second ratio (air pressure) + max (cycle stop)."""
    df = pd.read_csv(path)
    df["_time"] = pd.to_datetime(df["_time"], format="ISO8601")
    df = df.set_index("_time")

    out = pd.DataFrame()
    out["air_pressure_low_ratio"] = df["Air pressure low"].resample("1s").mean()
    out["cycle_stop_present"] = df["Cycle stop"].resample("1s").max().astype(int)
    return out.head(n_seconds).fillna(0).reset_index(drop=True)


def build_machine_frame(
    machine_id: str,
    cyclical: pd.DataFrame,
    acoustic: pd.DataFrame,
    flags: pd.DataFrame,
    grid: pd.DatetimeIndex,
    offsets: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Stitch the three aggregates onto a shared time grid, apply per-machine
    offsets, and compute heuristic OEE targets.
    """
    n = min(len(cyclical), len(acoustic), len(flags), len(grid))

    df = pd.DataFrame({
        "timestamp": grid[:n],
        "machine_id": machine_id,
        "setvelo_mean": cyclical["setvelo_mean"].iloc[:n].values
            + offsets.get("setvelo_shift", 0.0),
        "setpressure_cycle_count": cyclical["setpressure_cycle_count"].iloc[:n].values,
        "vibration": acoustic["vibration"].iloc[:n].values
            * offsets.get("vibration_scale", 1.0),
        "acoustic_emission": acoustic["acoustic_emission"].iloc[:n].values
            * offsets.get("ae_scale", 1.0),
        "air_pressure_low_ratio": flags["air_pressure_low_ratio"].iloc[:n].values,
        "cycle_stop_present": flags["cycle_stop_present"].iloc[:n].values,
        "cyclical_anomaly_score": cyclical["cyclical_anomaly_score"].iloc[:n].values,
        "acoustic_anomaly_score": acoustic["acoustic_anomaly_score"].iloc[:n].values,
    })

    # Heuristic targets — flags drag OEE down; quality wobbles slightly.
    df["availability"] = (1.0 - df["cycle_stop_present"].astype(float) * 0.3).clip(0, 1)
    df["performance"] = (1.0 - df["air_pressure_low_ratio"] * 0.4).clip(0, 1)
    df["quality"] = (0.97 + 0.03 * rng.random(len(df))).clip(0, 1)
    df["oee"] = df["availability"] * df["performance"] * df["quality"]

    return df[[
        "timestamp", "machine_id",
        "availability", "performance", "quality", "oee",
        "air_pressure_low_ratio", "cycle_stop_present",
        "vibration", "acoustic_emission",
        "setpressure_cycle_count", "setvelo_mean",
        "cyclical_anomaly_score", "acoustic_anomaly_score",
    ]]


def main() -> None:
    print(f"Reading sources from {DATA_DIR}")

    rng = np.random.default_rng(42)

    cyclical = aggregate_cyclical(
        DATA_DIR / "Clean_Data.csv",
        n_seconds=DURATION_SECONDS,
    )
    acoustic_dir = DATA_DIR / "Inkjet Printer Data Collection" / "Acoustic Sensor Data"
    acoustic = aggregate_acoustic(
        acoustic_dir / "machine_on_clean.csv",
        acoustic_dir / "machine_on_anomaly.csv",
        n_seconds=DURATION_SECONDS,
        rng=rng,
    )
    flags = aggregate_flags(
        DATA_DIR / "Inkjet Printer Data Collection" / "Air Pressure Low.csv",
        n_seconds=DURATION_SECONDS,
    )

    print(f"  cyclical rows: {len(cyclical)}")
    print(f"  acoustic rows: {len(acoustic)}")
    print(f"  flags rows:    {len(flags)}")

    grid = pd.date_range(ORIGIN_TIMESTAMP, periods=DURATION_SECONDS, freq="1s")

    inkjet_01 = build_machine_frame(
        "inkjet_01", cyclical, acoustic, flags, grid,
        offsets={},
        rng=rng,
    )
    inkjet_02 = build_machine_frame(
        "inkjet_02", cyclical, acoustic, flags, grid,
        offsets={"setvelo_shift": 0.5, "vibration_scale": 1.15, "ae_scale": 0.9},
        rng=rng,
    )

    out = (
        pd.concat([inkjet_01, inkjet_02], ignore_index=True)
        .sort_values(["timestamp", "machine_id"])
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(out)} rows to {OUTPUT_PATH}")
    print(f"  columns:    {list(out.columns)}")
    print(f"  time range: {out['timestamp'].min()} .. {out['timestamp'].max()}")
    print(f"  machines:   {sorted(out['machine_id'].unique())}")


if __name__ == "__main__":
    main()

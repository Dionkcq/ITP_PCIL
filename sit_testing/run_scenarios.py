"""Run Winardi's SIT test scenarios through the PCIL pipeline and grade them.

For each scenario recipe in systems/sit_scenarios/, this script:
  1. POSTs /pipeline/run to the orchestrator (default http://localhost:8000),
  2. saves the full response JSON (the evidence),
  3. grades the regression coefficients against the Expected Result from
     Winardi's workbook (hand-encoded below in EXPECTATIONS, so the grading
     is deterministic and reviewable),
  4. writes results.json + results.md summaries.

Grading vocabulary (mirrors how the workbook phrases expectations):
  dominant      the named feature must have the LARGEST |coefficient| of all
                features for that target (and match the expected sign, when
                the workbook states one).
  uniform_neg   every feature coefficient is negative and no single feature
                dominates (the workbook's "uniformly distributed small
                negative coefficients").
  zero          the named feature's coefficient is exactly ~0 (dead sensor).
  attributed    every named feature carries a meaningful share (>= 5%) of
                the target's total |coefficient| mass.
  observe       no mechanical pass/fail — the expectation is qualitative
                (collinearity / non-linearity); evidence is recorded for
                manual reading.

Coefficients are read from impacts.context[*].ranked_feature_impacts[*]
.raw_impact_score — the linear-regression weight on the MinMax-scaled
feature, which is what the workbook's "coefficient" expectations refer to.
(Scaling changes magnitudes, never signs or within-target ranking.)

Usage (host python, stdlib only):

    python sit_testing/run_scenarios.py
    python sit_testing/run_scenarios.py --base-url http://localhost:8000 --out sit_testing/results
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE_DIR = REPO_ROOT / "systems" / "sit_scenarios"

# Share of a target's total |coefficient| mass below which a feature is
# treated as "not attributed". 5% separates real weight from numerical dust.
ATTRIBUTION_SHARE = 0.05
# |coefficient| below this is "zero" (dead-sensor check). The zero-variance
# column MinMax-scales to a constant 0, so lstsq gives it weight 0 exactly;
# the epsilon only absorbs float noise.
ZERO_EPS = 1e-9

# ---------------------------------------------------------------------------
# Hand-encoded from 'SIT ITP Test Scenarios.xlsx' (Expected Result column).
# One entry per scenario: a list of checks. Where the workbook states a sign
# ("negative coefficient for X") it is encoded; where it only says
# "correlation with X", sign is None and only dominance/attribution is
# checked. Purely qualitative expectations become "observe" notes.
# ---------------------------------------------------------------------------
EXPECTATIONS: dict[int, list[dict]] = {
    1: [{"check": "dominant", "target": "line_oee_pct",
         "feature": "m1_downtime_min", "sign": -1}],
    2: [{"check": "dominant", "target": "line_availability_pct",
         "feature": "planned_changeover_min", "sign": -1}],
    3: [{"check": "uniform_neg", "target": "line_oee_pct"}],
    4: [{"check": "observe", "note": (
        "Collinearity stress test: m1 failure starves m2/m3, so the four "
        "features move together. The workbook predicts the un-regularized "
        "linear regression splits the weights erratically instead of "
        "blaming m1_mechanical_failure_min alone.")}],
    5: [{"check": "dominant", "target": "first_time_yield_pct",
         "feature": "bath_age_hours", "sign": -1}],
    6: [{"check": "dominant", "target": "scrap_rate_pct",
         "feature": "material_moisture_pct", "sign": +1}],
    7: [{"check": "dominant", "target": "part_surface_roughness_microns",
         "feature": "tool_strike_count", "sign": +1}],
    8: [{"check": "attributed", "target": "defect_density_scale",
         "features": ["ambient_humidity_pct", "hydraulic_pressure_bar"]}],
    9: [{"check": "dominant", "target": "line_throughput_units_hr",
         "feature": "actual_conveyor_rpm", "sign": +1}],
    10: [{"check": "dominant", "target": "average_cycle_time_sec",
          "feature": "is_trainee_operator", "sign": +1}],
    11: [{"check": "dominant", "target": "strokes_per_minute",
          "feature": "oil_temperature_c", "sign": -1}],
    12: [{"check": "dominant", "target": "daily_kwh_energy_consumption",
          "feature": "air_compressor_duty_cycle_pct", "sign": +1},
         {"check": "observe", "note": (
             "The point of the scenario: compressor duty stays high while "
             "production drops to zero on weekends, so energy should be "
             "attributed to the compressor, not production volume.")}],
    13: [{"check": "attributed", "target": "electricity_cost_per_hour",
          "features": ["furnace_1_active", "furnace_2_active",
                       "is_peak_tariff_window"]},
         {"check": "attributed", "target": "total_kw_draw",
          "features": ["furnace_1_active", "furnace_2_active"]}],
    14: [{"check": "dominant", "target": "motor_vibration_mm_s",
          "feature": "lubrication_level_pct", "sign": -1}],
    15: [{"check": "observe", "note": (
        "Non-linearity stress test: flow collapses exponentially once the "
        "filter saturates. The workbook asks whether the LINEAR baseline "
        "copes or falls short — read the coefficients as a straight-line "
        "approximation of a curve.")}],
    16: [{"check": "dominant", "target": "assembly_station_idle_min",
          "feature": "agv_in_charging_bay_count", "sign": +1}],
    17: [{"check": "dominant", "target": "defective_parts_per_1k",
          "feature": "is_supplier_b", "sign": +1}],
    18: [{"check": "zero", "target": "extruder_thickness_error_mm",
          "feature": "heater_1_temp_sensor"}],
    19: [{"check": "attributed", "target": "defect_rate_pct",
          "features": ["is_line_1", "product_complexity_scale"]},
         {"check": "observe", "note": (
             "Confounder test: Line 1 only looks bad because it gets the "
             "complex products. Ideally product_complexity_scale carries "
             "more weight than is_line_1.")}],
    20: [{"check": "dominant", "target": "palletizer_cycle_time_sec",
          "feature": "gripper_wear_index", "sign": None}],
}


def coef_table(response: dict) -> dict[str, dict[str, float]]:
    """{target: {feature: raw coefficient}} from a /pipeline/run response."""
    out: dict[str, dict[str, float]] = {}
    for ctx in response["impacts"]["context"]:
        out[ctx["target"]] = {
            fi["feature"]: fi["raw_impact_score"]
            for fi in ctx["ranked_feature_impacts"]
        }
    return out


def run_check(check: dict, coefs: dict[str, dict[str, float]]) -> dict:
    kind = check["check"]
    if kind == "observe":
        return {"check": check, "verdict": "OBSERVE", "detail": check["note"]}

    target = check["target"]
    if target not in coefs:
        return {"check": check, "verdict": "FAIL",
                "detail": f"target {target} missing from impacts"}
    weights = coefs[target]
    total_abs = sum(abs(c) for c in weights.values()) or 1.0

    if kind == "dominant":
        feat, sign = check["feature"], check["sign"]
        coef = weights.get(feat)
        if coef is None:
            return {"check": check, "verdict": "FAIL",
                    "detail": f"{feat} missing from impacts"}
        top = max(weights, key=lambda f: abs(weights[f]))
        ok = top == feat and (sign is None or coef * sign > 0)
        return {"check": check, "verdict": "PASS" if ok else "FAIL",
                "detail": (f"{feat} coef={coef:+.4f} "
                           f"({abs(coef) / total_abs:.0%} of |coef| mass); "
                           f"largest |coef| is {top}")}

    if kind == "uniform_neg":
        all_neg = all(c < 0 for c in weights.values())
        shares = {f: abs(c) / total_abs for f, c in weights.items()}
        dominated = max(shares.values()) >= 0.5  # one feature holding half+
        ok = all_neg and not dominated
        detail = ", ".join(f"{f}={weights[f]:+.4f} ({shares[f]:.0%})"
                           for f in weights)
        return {"check": check, "verdict": "PASS" if ok else "FAIL",
                "detail": detail}

    if kind == "zero":
        coef = weights.get(check["feature"], None)
        ok = coef is not None and abs(coef) < ZERO_EPS
        return {"check": check, "verdict": "PASS" if ok else "FAIL",
                "detail": f"{check['feature']} coef={coef!r}"}

    if kind == "attributed":
        shares = {f: abs(weights.get(f, 0.0)) / total_abs
                  for f in check["features"]}
        ok = all(s >= ATTRIBUTION_SHARE for s in shares.values())
        detail = ", ".join(f"{f}: {s:.0%}" for f, s in shares.items())
        return {"check": check, "verdict": "PASS" if ok else "FAIL",
                "detail": detail}

    raise ValueError(f"unknown check kind {kind!r}")


def scenario_verdict(check_results: list[dict]) -> str:
    """FAIL if any mechanical check fails; OBSERVE only when nothing
    mechanical exists; PASS otherwise."""
    verdicts = [c["verdict"] for c in check_results]
    if "FAIL" in verdicts:
        return "FAIL"
    if "PASS" in verdicts:
        return "PASS"
    return "OBSERVE"


def post_pipeline_run(base_url: str, config_path: str) -> dict:
    req = urllib.request.Request(
        f"{base_url}/pipeline/run",
        data=json.dumps({"config_path": config_path}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "sit_testing" / "results")
    args = ap.parse_args()

    runs_dir = args.out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "scenarios": [],
    }

    for sn in sorted(EXPECTATIONS):
        recipe = RECIPE_DIR / f"scenario_{sn}.yaml"
        if not recipe.exists():
            print(f"scenario_{sn}: recipe missing, skipped")
            continue
        config_path = f"systems/sit_scenarios/scenario_{sn}.yaml"
        ran_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            response = post_pipeline_run(args.base_url, config_path)
        except Exception as exc:  # noqa: BLE001 - recorded, run continues
            entry = {"scenario": sn, "ran_at": ran_at, "verdict": "ERROR",
                     "error": f"{type(exc).__name__}: {exc}"}
            summary["scenarios"].append(entry)
            print(f"scenario_{sn}: ERROR {entry['error']}")
            continue

        (runs_dir / f"scenario_{sn}.json").write_text(
            json.dumps(response, indent=2), encoding="utf-8")

        coefs = coef_table(response)
        checks = [run_check(c, coefs) for c in EXPECTATIONS[sn]]
        entry = {
            "scenario": sn,
            "ran_at": ran_at,
            "input_rows": response["input_rows"],
            "recommendation_status": response["recommendation_status"],
            "verdict": scenario_verdict(checks),
            "checks": checks,
            "coefficients": coefs,
        }
        summary["scenarios"].append(entry)
        print(f"scenario_{sn}: {entry['verdict']}")

    summary["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tally = {}
    for e in summary["scenarios"]:
        tally[e["verdict"]] = tally.get(e["verdict"], 0) + 1
    summary["tally"] = tally

    (args.out / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown summary table for quick reading / pasting into Teams.
    lines = [
        "# SIT scenario results", "",
        f"Run: {summary['started_at']} -> {summary['finished_at']}  "
        f"({args.base_url})", "",
        f"Tally: {tally}", "",
        "| # | Verdict | Evidence |",
        "|---|---------|----------|",
    ]
    for e in summary["scenarios"]:
        if e["verdict"] == "ERROR":
            lines.append(f"| {e['scenario']} | ERROR | {e['error']} |")
            continue
        evid = "<br>".join(f"[{c['verdict']}] {c['detail']}" for c in e["checks"])
        lines.append(f"| {e['scenario']} | {e['verdict']} | {evid} |")
    (args.out / "results.md").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")
    print(f"\nTally: {tally}")
    print(f"Wrote {args.out / 'results.json'} and results.md; "
          f"raw responses in {runs_dir}/")


if __name__ == "__main__":
    main()

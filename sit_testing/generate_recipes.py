"""Generate PCIL config recipes from Winardi's SIT test-scenario workbook.

Reads `SIT ITP Test Scenarios.xlsx` (one row per scenario: S/N, Target,
Feature, Expected Result) and writes one recipe YAML per scenario into
`systems/sit_scenarios/`. Each recipe points the pipeline at the matching
`scenario_<n>` PostgreSQL table (restored from `itp_test_scenarios.sql`).

Usage (from the repo root, host python with openpyxl installed):

    python sit_testing/generate_recipes.py
    python sit_testing/generate_recipes.py --xlsx "path/to/SIT ITP Test Scenarios.xlsx"

Re-runnable: when Winardi extends the workbook to 30 scenarios, run it
again and the new recipes appear alongside the old ones.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Dion's ITP folder layout keeps Winardi's files outside the repo, in a
# sibling data/ folder. Override with --xlsx for any other layout.
DEFAULT_XLSX = REPO_ROOT.parent / "data" / "testing" / "SIT ITP Test Scenarios.xlsx"
DEFAULT_OUT = REPO_ROOT / "systems" / "sit_scenarios"

# Column-name suffix -> unit hint, so the generated feature descriptions
# read as English instead of raw snake_case. Purely cosmetic: the LLM
# prompt uses these lines, the regression does not.
_UNIT_HINTS = {
    "_min": "minutes",
    "_sec": "seconds",
    "_hours": "hours",
    "_pct": "percent",
    "_c": "degrees Celsius",
    "_rpm": "RPM",
    "_bar": "bar",
    "_lpm": "litres per minute",
    "_gpm": "gallons per minute",
    "_kwh": "kWh",
    "_psi": "PSI",
    "_amps": "amperes",
    "_mm_s": "mm/s",
    "_mm": "millimetres",
    "_grams": "grams",
    "_microns": "microns",
    "_mps": "metres per second",
    "_count": "count",
}


def describe(column: str) -> str:
    """Best-effort plain-English one-liner for a scenario column name."""
    if column.startswith("is_"):
        return f"Binary flag (0 or 1): {column[3:].replace('_', ' ')}."
    for suffix, unit in _UNIT_HINTS.items():
        if column.endswith(suffix):
            stem = column[: -len(suffix)].replace("_", " ")
            return f"{stem.capitalize()} ({unit}) over the window."
    return f"{column.replace('_', ' ').capitalize()} over the window."


def load_rows(xlsx: Path) -> list[dict]:
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "openpyxl is required: python -m pip install openpyxl"
        ) from exc
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    rows = []
    for row in wb.active.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        rows.append(
            {
                "sn": int(row[0]),
                "targets": [t.strip() for t in str(row[1]).split(",") if t.strip()],
                "features": [f.strip() for f in str(row[2]).split(",") if f.strip()],
                "expected": " ".join(str(row[3]).split()),
            }
        )
    return rows


def recipe_yaml(row: dict) -> str:
    """Render one scenario recipe. Plain string assembly (not yaml.dump) so
    the file keeps the same commented, sectioned shape as the hand-written
    recipes in systems/."""
    sn = row["sn"]
    lines = [
        "# ============================================================",
        f"# SIT ITP Test Scenario {sn} (Winardi, 2026-07-06) — GENERATED",
        f"# by sit_testing/generate_recipes.py from 'SIT ITP Test Scenarios.xlsx'.",
        "# Expected result (verbatim from the workbook):",
    ]
    # Wrap the expected-result prose at ~70 chars so the header stays readable.
    words, line = row["expected"].split(), "#  "
    for w in words:
        if len(line) + len(w) + 1 > 72:
            lines.append(line)
            line = "#  "
        line += " " + w
    lines.append(line)
    lines += [
        f"# Data: table scenario_{sn} (restored from itp_test_scenarios.sql).",
        "# ============================================================",
        "",
        "system: sit_scenarios",
        "",
        "pipeline:",
        '  output_dir: "output"',
        "",
        "trigger:",
        "  source_type: postgres",
        f"  table: scenario_{sn}",
        '  mode: "all"',
        "  start_time: null",
        "  end_time: null",
        "  last_n: null",
        "",
        "input:",
        "  timestamp_column: timestamp",
        "",
        "  numerical_features:",
    ]
    lines += [f"    - {f}" for f in row["features"]]
    lines += ["", "  categorical_features: []", "", "  targets:"]
    lines += [f"    - {t}" for t in row["targets"]]
    lines += ["", "feature_descriptions:"]
    lines += [f'  {f}: "{describe(f)}"' for f in row["features"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = load_rows(args.xlsx)
    args.out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = args.out / f"scenario_{row['sn']}.yaml"
        path.write_text(recipe_yaml(row), encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(REPO_ROOT)}  "
              f"(targets={row['targets']}, {len(row['features'])} features)")
    print(f"{len(rows)} recipes written to {args.out}")


if __name__ == "__main__":
    main()

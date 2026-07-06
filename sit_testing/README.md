# SIT scenario testing

Runs Winardi's SIT test scenarios (July 2026) through the PCIL pipeline and
grades the regression coefficients against his expected results.

Inputs (provided by Winardi, kept outside the repo in `../data/testing/`):

- `itp_test_scenarios.sql` — a **pg_dump custom-format archive** (made with
  PostgreSQL 17.6, despite the `.sql` name) holding one table per scenario
  (`scenario_1` … `scenario_20`), 1000 rows each: a `timestamp` column plus
  the scenario's target(s) and features.
- `SIT ITP Test Scenarios.xlsx` — one row per scenario: Target, Feature
  list, and the Expected Result prose.

## How to run

1. **Start the stack** (from the repo root):

   ```bash
   docker compose up -d
   ```

2. **Restore the scenario tables** into the bundled postgres. The stack's
   `pgvector:pg16` image cannot read a PG17 dump, so use a one-shot
   `postgres:17` client container on the compose network:

   ```bash
   docker run --rm --network pcil_pg_default \
     -e PGPASSWORD=<POSTGRES_PASSWORD from .env> \
     -v "<abs path to data/testing>:/dump:ro" \
     postgres:17 pg_restore -h postgres -U pcil -d pcil \
     --no-owner --no-privileges /dump/itp_test_scenarios.sql
   ```

   One ignorable error is expected (`SET transaction_timeout` is a
   PG17-only setting the PG16 server rejects). Verify with:

   ```bash
   docker exec pcil-postgres psql -U pcil -d pcil \
     -c "\dt scenario_*"
   ```

3. **Generate the recipes** (one config YAML per scenario, written to
   `systems/sit_scenarios/`; needs `pip install openpyxl`):

   ```bash
   python sit_testing/generate_recipes.py
   ```

4. **Run + grade all scenarios** (stdlib only):

   ```bash
   python sit_testing/run_scenarios.py --out sit_testing/results
   ```

   Outputs: `results.json` (verdicts + coefficients + timestamps),
   `results.md` (summary table), `runs/scenario_<n>.json` (full
   `/pipeline/run` responses — the raw evidence).

Both scripts are re-runnable: when the workbook grows to 30 scenarios,
re-run step 3 (new recipes appear), add the new expectations to
`EXPECTATIONS` in `run_scenarios.py`, and re-run step 4.

## How grading works

`run_scenarios.py` hand-encodes each Expected Result as mechanical checks
on `raw_impact_score` (the linear-regression coefficient on the MinMax-
scaled feature — scaling changes magnitudes, never signs or within-target
ranking): `dominant` (largest |coef|, with the stated sign), `uniform_neg`,
`zero`, `attributed` (>= 5% of the target's |coef| mass), or `observe`
(qualitative stress tests — evidence recorded, no auto verdict).

## Data notes found while testing (worth confirming with Winardi)

- `scenario_8` has a `room_temp_c` column the workbook does not list as a
  feature; the recipe follows the workbook and omits it.
- `scenario_13.furnace_1_active` is constant (always 1) and
  `furnace_2_active` is identical to `is_peak_tariff_window` (r = 1.000) —
  no correlational model can attribute weight to a constant, and duplicate
  columns share weight arbitrarily.
- `scenario_14.motor_rpm` is constant (always 1500).
- `scenario_11.oil_temperature_c` and `ambient_temp_c` are perfectly
  correlated (r = 1.000).

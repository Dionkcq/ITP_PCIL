# PCIL Job Orchestrator — Deployment & Testing Guide

How to run the PCIL solution as a Docker container and call its API.
Written for the SIMTech test environment; everything runs from one
container via `docker-compose.yml`.

The container serves two things on **port 8000**:

- the **REST API** (FastAPI) — the programmatic way to trigger the solution
- the **operator dashboard** (browser UI) at `/dashboard/` — the point-and-click way

---

The compose stack also starts PostgreSQL with pgvector. RAG recovery
records are ingested from `data/RAG/*.docx`, embedded with BAAI/bge-m3,
stored in PostgreSQL, and retrieved with BM25 + pgvector + RRF before
Gemini writes the final recommendation.

## 1. What you need

| Item | Notes |
|---|---|
| Docker with Compose v2 | `docker compose version` should print v2.x |
| This repository (or just `docker-compose.yml`) | https://github.com/Dionkcq/ITP_PCIL |
| The `data/` folder | Provided separately by the team — **not** in git. Contains the trained model bundles, sample shop-floor CSV, and RAG recovery documents. |
| `GEMINI_API_KEY` (optional) | Needed for live LLM-written recommendations. Without it (or without outbound internet) everything still works, but `operator_recommendation` is a fixed fallback string instead of generated text. |

Expected layout before starting:

```
<deploy folder>/
├── docker-compose.yml
├── .env                     # optional — see step 3
├── systems/                # config recipes (in the repo; or provided with data/)
│   └── inkjet_printer/
│       └── config.yaml              # the pipeline "recipe" — editable, see section 7
└── data/                    # provided by the team
    ├── mock_shop_floor.csv          # sample shop-floor slice (config.yaml points here)
    ├── cyclical_inkjet_01.pkl       # trained cyclical anomaly bundle
    ├── non_cyclical_inkjet_01.pkl   # trained non-cyclical anomaly bundle
    └── RAG/                         # recovery-knowledge DOCX documents
        └── *.docx
```

A repo clone already contains `systems/`; only `data/` has to be added.
If either folder lives somewhere else, set `PCIL_DATA_DIR=<path>` /
`PCIL_SYSTEMS_DIR=<path>` in `.env` instead of moving it.

## 2. Get the image

Pick whichever fits the test environment:

**A. Pull the prebuilt image** (needs internet):

```bash
docker compose pull
```

**B. Build from source** (needs internet + a full repo checkout — the
build compiles the dashboard with Node and installs the Python stack):

```bash
docker compose build
```

**C. Load from a tarball** (offline environments — ask the team for
`pcil_image.tar`):

```bash
docker load -i pcil_image.tar
```

## 3. Configure (optional)

Create a file named `.env` next to `docker-compose.yml`:

```bash
GEMINI_API_KEY=your_real_key_here
POSTGRES_PASSWORD=
# PCIL_PORT=8000          # change if 8000 is taken on the host
# PCIL_DATA_DIR=./data    # change if data/ lives elsewhere
```

Skipping `GEMINI_API_KEY` is fine — the service boots without it and
degrades gracefully (see section 7).

`POSTGRES_PASSWORD` is required for the private pgvector service. Generate
a local value, for example with PowerShell:

```powershell
[guid]::NewGuid().ToString("N")
```

## 4. Start and verify

```bash
docker compose up -d
docker compose ps          # state should be "running (healthy)" after ~20 s
curl http://localhost:8000/
```

On first startup, the API runs RAG migrations and ingests changed DOCX
files when `RAG_BACKEND=postgres` and `RAG_AUTO_INGEST=true` (the compose
defaults). BGE-M3 model files are cached in a Docker volume, so the first
boot can be slower than later restarts. If you add or edit DOCX recovery
files after startup, refresh the PostgreSQL RAG index without rebuilding:

```bash
curl -X POST http://localhost:8000/rag/reindex
```

`GET /` returns service metadata including an endpoint index. Then open
in a browser (replace `localhost` with the host's IP when testing from
another machine on the LAN):

- `http://localhost:8000/dashboard/` — operator dashboard
- `http://localhost:8000/docs` — interactive Swagger UI (try any endpoint from the browser)

Logs and shutdown:

```bash
docker compose logs -f     # follow the orchestrator logs
docker compose down        # stop and remove the container
```

## 5. Triggering the solution

There is no cron or watcher inside the container — **the trigger IS the
API call**. Whatever system decides "diagnose now" (a test script, an
ingestion job, a human) sends one HTTP request and gets the full
diagnosis back in the response.

### 5a. Run the pipeline on the configured source

Uses the slice recipe baked into `systems/inkjet_printer/config.yaml`
(which points at `data/mock_shop_floor.csv` inside the mounted data
folder):

```bash
curl -X POST http://localhost:8000/pipeline/run \
     -H "Content-Type: application/json" \
     -d '{"config_path": "systems/inkjet_printer/config.yaml", "persist": false}'
```

### 5b. Run the pipeline on an uploaded CSV

Same pipeline, but the shop-floor slice arrives as a file upload — no
config editing needed. The CSV must match the schema in `config.yaml`'s
`input` block (timestamp column, numerical/categorical features,
targets):

```bash
curl -X POST http://localhost:8000/pipeline/run_csv \
     -F "config_path=systems/inkjet_printer/config.yaml" \
     -F "persist=false" \
     -F "file=@shop_floor_slice.csv"
```

### 5c. Run the pipeline from the dockerized PostgreSQL shop-floor table

The compose stack includes a PostgreSQL service. To pull the slice from the
database instead of a CSV, use the postgres recipe
`systems/inkjet_printer/config_postgres.yaml` (it sets `source_type: postgres`
and `table: shop_floor`). On startup the app seeds that table from the mounted
`mock_shop_floor.csv` — idempotent, so it is skipped once the table has rows;
set `PCIL_SEED_SHOP_FLOOR=false` to disable, or re-seed manually with
`POST /shopfloor/seed`. Then:

```bash
curl -X POST http://localhost:8000/pipeline/run \
     -H "Content-Type: application/json" \
     -d '{"config_path": "systems/inkjet_printer/config_postgres.yaml", "persist": false}'
```

The slice mode maps to SQL (`all` → `ORDER BY timestamp`; `time_range` →
`WHERE timestamp BETWEEN`; `last_n` → `ORDER BY timestamp DESC LIMIT`).
Everything downstream (preprocess → impacts → RAG → recommendation) is
identical to the CSV path.

### Response shape (both variants)

```json
{
  "status": "ok",
  "input_rows": 625,
  "golden_rows": 625,
  "impacts": { "...": "ranked feature impacts per target (Week-3 JSON schema)" },
  "target_summary": { "availability": 1.0, "performance": 0.957, "quality": 0.985, "oee": 0.943 },
  "recovery_records": [ { "error": "...", "cause": "...", "recovery": "...", "source_doc": "..." } ],
  "operator_recommendation": "LLM-composed paragraph for the operator",
  "artifacts": {}
}
```

- `impacts` — which input features drove each target up or down in this
  context window, ranked, with raw + standardized scores.
- `recovery_records` — top-3 matching entries retrieved from the DOCX
  recovery documents (local TF-IDF retrieval, no internet needed).
- `operator_recommendation` — Gemini-composed plain-language guidance;
  falls back to a fixed string when no key/internet is available.

The same flow is available point-and-click in the dashboard's
**Diagnosis** tab (configured source or CSV upload).

## 6. API reference

Base URL: `http://<host>:8000`

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service metadata + endpoint index (use as health check) |
| POST | `/pipeline/run` | Full pipeline on the slice configured in `config.yaml` |
| POST | `/pipeline/run_csv` | Full pipeline on an uploaded shop-floor CSV (multipart) |
| POST | `/pipeline/save_csv` | Write the configured slice to disk as `context_window_<start>_<end>.csv` |
| POST | `/anomaly/train` | Train an anomaly model from uploaded CSV(s); saves a `.pkl` bundle into `data/` |
| POST | `/anomaly/score` | Score time-series rows against a trained bundle |
| GET | `/configs` | List the available config recipes |
| GET | `/configs/load` | Load a recipe as structured data |
| POST | `/configs/validate` | Dry-run validation of an edited recipe (nothing written) |
| POST | `/configs/save` | Validate + save a recipe (timestamped backup kept on overwrite) |
| POST | `/configs/create` | Create a brand-new system folder + recipe (never overwrites) |
| POST | `/configs/delete` | Delete a recipe (recoverable — moved into `.backups/`, not destroyed) |
| GET | `/anomaly/models` | List the trained `.pkl` bundles present in `data/` |
| POST | `/rag/reindex` | Re-ingest DOCX recovery records into PostgreSQL and refresh BM25/vector RAG |
| GET | `/docs` | Swagger UI — interactive docs for every endpoint above |
| GET | `/dashboard/` | Operator dashboard (Diagnosis / Anomaly check / Train / Config recipes tabs) |

### Anomaly scoring example

`model_type` is one of `cyclical`, `non_cyclical`, `irregular`; a
matching bundle `data/<model_type>_<model_id>.pkl` must exist (the
provided `data/` folder ships `cyclical_inkjet_01` and
`non_cyclical_inkjet_01`):

```bash
curl -X POST http://localhost:8000/anomaly/score \
     -H "Content-Type: application/json" \
     -d '{
       "model_type": "cyclical",
       "model_id": "inkjet_01",
       "data": [
         {"timestamp": "2026-06-15T09:00:00.000", "machine_id": "inkjet_01", "signal_value": 0.42},
         {"timestamp": "2026-06-15T09:00:00.001", "machine_id": "inkjet_01", "signal_value": 0.45}
       ]
     }'
```

Cyclical and irregular responses include per-cycle/window
`anomaly_scores` plus `is_anomaly` flags computed against the
`threshold` stored at train time. Non-cyclical scores are class
probabilities from a supervised Random Forest — threshold directly
(e.g. at 0.5). The API never writes to the shop-floor database; the
caller decides which DB row a score belongs to.

Training examples for all three model types are in the repo
`README.md` ("Factory test API endpoints") and runnable interactively
from `/docs`.

## 7. Changing the pipeline recipe (e.g. a new sensor column)

`systems/inkjet_printer/config.yaml` is the pipeline's recipe: where the
shop-floor slice comes from, how to slice it (all rows / time range /
last N), which columns are features and targets. Because the folder is
mounted from the host, **recipe changes need no image rebuild and no
container restart** — the next API call uses the saved version.

Two ways to change it:

- **Dashboard (recommended):** the **Config recipes** tab edits the recipe
  as a form — add/remove feature columns with descriptions, change targets
  or the slice mode. Every save is validated server-side first (an invalid
  recipe is rejected with the reasons listed, the file is never corrupted)
  and the previous version is backed up to `systems/<system>/.backups/`.
  "Save as new" creates a separate recipe (e.g. `config_test2.yaml`) that
  becomes selectable in the Diagnosis tab without touching the original.
  The tab's **New system** section creates a whole new system folder
  (`systems/<name>/<recipe>.yaml`) from the current form — onboard a
  second system without touching the repository at all. One system can
  hold several recipes for different purposes ("Save as new"); deleting a
  recipe moves it into `.backups/` rather than destroying it, so a wrong
  click is recoverable from disk.
- **Text editor:** edit the YAML on the host directly. The same validation
  runs when the recipe is used, but there is no backup — the dashboard
  path is safer.

Note: the testing CSV (uploaded or configured) must contain the columns
the recipe names. If a pipeline run reports a missing column, the recipe
and the data disagree — fix whichever is wrong.

## 8. Degraded modes (by design)

| Situation | Behaviour |
|---|---|
| No `GEMINI_API_KEY` or no outbound internet | Pipeline still returns impacts + recovery records; `operator_recommendation` is a fallback string. LLM calls time out after 30 s. |
| `data/RAG/` missing | `recovery_records` comes back empty; rest of the response unaffected. |
| Requested `.pkl` bundle missing | `/anomaly/score` returns 404 listing the paths it tried. |
| Empty/short input data | Scoring returns zero cycles/windows rather than erroring. |

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `trigger.source not found: ...` from `/pipeline/run` | `data/` folder not mounted or missing `mock_shop_floor.csv`. Check `PCIL_DATA_DIR` and `docker compose config` to see the resolved mount. |
| `/dashboard/` returns 404 | Image was built without the dashboard stage — rebuild with `docker compose build` (the provided image includes it). |
| Port already in use | Set `PCIL_PORT=<other>` in `.env`, rerun `docker compose up -d`. |
| Container unhealthy | `docker compose logs pcil` — the orchestrator prints the failing stage. |
| Can't reach API from another machine | Use the Docker host's LAN IP, and check the host firewall allows inbound on the chosen port. |

---

Maintainer notes (image publishing, dev setup, tests) live in
`README.md`. Contact: Dion Ko (ITP Team 21).

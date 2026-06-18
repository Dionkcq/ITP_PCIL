# PCIL Architecture (C4 model)

This document describes the PCIL architecture using the [C4 model](https://c4model.com)
— a set of diagrams at increasing zoom levels:

1. **Context** — the system as one box, its users, and the external systems it talks to.
2. **Container** — the deployable/runnable pieces inside PCIL (apps, services, data stores).
3. **Component** — the major code modules inside one container.
4. *Code* — class/function level. **Skipped** on purpose (the source is the truth at that zoom).

Plus a **runtime data-flow** diagram showing the contract handed between pipeline stages.

> **Scope.** This reflects the **current `main` branch** (commit `49a2696`): a CSV/file
> data source and a **single** orchestrator container (pipeline + anomaly together). The
> PostgreSQL source and the pipeline/anomaly container split are planned on a separate
> branch — see [Roadmap](#roadmap-not-yet-on-main). Levels 1 (Context) and 2 (Container)
> are shown in the [README](README.md#architecture-c4); this document covers Level 3
> (Component) and the data-flow.

All diagrams are Mermaid and render natively on GitHub.

---

## Level 3 — Component (PCIL Job Orchestrator)

Inside the single FastAPI container, the request coordinator (`orchestrator.py`) wires
together the pipeline stages, the anomaly subpackages, and the config-recipe manager.
Data passes between stages **in memory**; nothing is written to disk during a normal run.

```mermaid
C4Component
    title Component diagram — PCIL Job Orchestrator (FastAPI)

    Person(operator, "Operator / engineer", "")
    System_Ext(gemini, "Google Gemini API", "LLM (gemini-2.5-flash)")
    ContainerDb(recipes, "Config recipes", "YAML files (systems/)")
    ContainerDb(rtdata, "Runtime data", "Files (data/): CSV slice, RAG .docx, .pkl bundles")

    Container_Boundary(orch, "PCIL Job Orchestrator") {
        Component(api, "API + coordinator", "orchestrator.py / FastAPI", "Routes for /pipeline, /anomaly, /configs; wires the stages in memory")
        Component(trigger, "Trigger / slice", "trigger.py", "Selects the window: all / time_range / last_n")
        Component(prep, "Preprocess (Pipeline 1)", "preprocess.py / scikit-learn", "sklearn Pipeline + ColumnTransformer to the Golden DataFrame")
        Component(adapter, "Adapter", "adapter.py", "Golden DataFrame to X / y arrays; validates the 0-1 range")
        Component(model, "Context model (Pipeline 2)", "train_context_model.py", "Multi-target LinearRegression to per-target ranked impacts")
        Component(rag, "RAG retrieval", "rag/loader.py + rag/lookup.py", "DOCX to records; TF-IDF + cosine top-k")
        Component(composer, "LLM composer (Pipeline 3)", "rag/composer.py / google-genai", "Builds the grounded prompt and calls Gemini")
        Component(anomaly, "Anomaly detection", "utils/anomaly/ (base + cyclical / non_cyclical / irregular)", "Per-data-type score/train over AnomalyModel + PerMachineNormaliser; torch for the cyclical autoencoder")
        Component(cfg, "Config recipe manager", "orchestrator.py helpers", "Validate / save / create / delete recipes; path-traversal safe")
    }

    Rel(operator, api, "HTTP requests", "JSON")
    Rel(api, trigger, "Pull slice")
    Rel(trigger, rtdata, "Reads CSV slice")
    Rel(api, prep, "Preprocess slice")
    Rel(prep, adapter, "Golden DataFrame")
    Rel(adapter, model, "X / y arrays")
    Rel(api, rag, "Retrieve recovery records")
    Rel(rag, rtdata, "Reads RAG .docx")
    Rel(api, composer, "Compose recommendation")
    Rel(composer, gemini, "Prompt", "HTTPS")
    Rel(api, anomaly, "Score / train")
    Rel(anomaly, rtdata, "Reads / writes .pkl bundles")
    Rel(api, cfg, "Manage recipes")
    Rel(cfg, recipes, "Reads / writes YAML")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

Notes:
- The **dashboard** (a separate container, see README) is also served by this process via
  FastAPI `StaticFiles` at `/dashboard`, but it is not a code component of the pipeline.
- The optional Flask `rag_frontend/` demo UI is **not** part of the deployed image and is
  omitted here.

---

## Runtime data-flow — the stage contracts

The diagnosis path (`POST /pipeline/run`). Each arrow is the contract the previous stage
produces and the next one accepts — "stage 1 produces this, stage 2 consumes it":

```mermaid
flowchart TD
    req["POST /pipeline/run<br/>input: config recipe path"]
    slice["Trigger / _pull_slice<br/>contract: shop-floor slice DataFrame<br/>(timestamp + declared feature/target columns)"]
    golden["Preprocess (Pipeline 1)<br/>contract: Golden DataFrame<br/>(timestamp, targets passthrough, features scaled 0-1)"]
    xy["Adapter<br/>contract: X (rows x features), y (rows x targets)"]
    impacts["Context model (Pipeline 2)<br/>contract: impacts dict<br/>(per-target ranked feature impacts)<br/>+ target_summary (window means)"]
    records["RAG retrieval<br/>contract: top-k recovery records<br/>(error, cause, recovery, source_doc)"]
    reco["LLM composer (Pipeline 3)<br/>contract: operator_recommendation<br/>+ recommendation_status"]
    resp["JSON response<br/>impacts + target_summary + records<br/>+ recommendation + status"]

    req --> slice --> golden --> xy --> impacts --> records --> reco --> resp
```

`target_summary` (from the context-model stage) is also fed into the composer, so the
recommendation is grounded in measured performance, and it drives the dashboard KPI cards.

### Anomaly scoring (separate engineer-facing API)

Anomaly detection is **input to output only** — PCIL never writes to the shop-floor data.
The engineer calls the API and writes the returned score back themselves (only they know
the row mapping):

```mermaid
flowchart LR
    raw["Engineer: POST /anomaly/score<br/>input: raw time-series rows + model_type"]
    pipe["Anomaly subpackage<br/>slice / features / per-machine standardise / model"]
    score["contract: anomaly_score per cycle or window<br/>(+ threshold, is_anomaly)"]
    back["Engineer writes the score back into<br/>their shop-floor data themselves"]

    raw --> pipe --> score --> back
```

### Contract table

| Stage | Input | Output contract |
|---|---|---|
| Trigger (`_pull_slice`) | config recipe (trigger mode + source) | shop-floor slice DataFrame (timestamp + declared columns) |
| Preprocess (Pipeline 1) | slice + recipe schema | Golden DataFrame: timestamp, targets (passthrough), features scaled 0-1 |
| Adapter | Golden DataFrame | `X` (rows x features), `y` (rows x targets), names; raises if features leave 0-1 |
| Context model (Pipeline 2) | `X`, `y` | impacts dict (`system`, `context_window`, per-target `ranked_feature_impacts`) + fitted model; orchestrator also computes `target_summary` |
| RAG retrieval | query from target names + top feature descriptions | top-k records (`error`, `cause`, `recovery`, `source_doc`) |
| LLM composer (Pipeline 3) | impacts + records + `target_summary` | `operator_recommendation` + `recommendation_status` (ok / no_records / retrieval_failed / llm_unavailable / rag_unavailable) |
| Response | all of the above | JSON: impacts, target_summary, recovery_records, operator_recommendation, recommendation_status, artifacts |
| `/anomaly/score` | raw time-series rows + `model_type` (+ `model_id`) | `anomaly_score` per cycle/window (+ `threshold`, `is_anomaly`) |
| `/anomaly/train` | uploaded CSV(s) + params | persisted `.pkl` bundle under `data/` |

---

## Roadmap (not yet on `main`)

These change the diagrams above and are deliberately excluded so this doc matches what
ships on `main` today. They are being built on a separate branch:

- **PostgreSQL source (P1).** The Context view gains an external **Shop-floor PostgreSQL
  database**; the orchestrator's `_pull_slice` gains a SQL branch (query built from the
  recipe; `psycopg` + SQLAlchemy via a `PCIL_DB_URL` env var). The CSV path stays as a
  fallback.
- **Pipeline / anomaly container split (P2).** The single orchestrator container becomes
  **two** containers — a lightweight **Pipeline service** and an **Anomaly service** (with
  `torch` only in the anomaly image) — so projects that do not need anomaly detection can
  run the pipeline alone.

When that branch merges, add a second Container diagram here showing both services + the
database, and keep this one as the "single-container" baseline for comparison.
```

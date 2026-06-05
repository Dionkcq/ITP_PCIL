# PCIL Operator Dashboard (React + Vite)

A thin React client for the PCIL Job Orchestrator. It holds **no** pipeline
logic — it calls the API and renders the JSON. Three tabs:

- **Diagnosis** — run the pipeline on the configured source *or* upload a
  shop-floor CSV (`/pipeline/run`, `/pipeline/run_csv`). Shows KPI cards, the
  LLM operator recommendation, a ranked bar chart of what is driving each
  target, and the recovery evidence. Download the result as JSON, print it to
  PDF, export the context-window CSV (`/pipeline/save_csv`), and re-view past
  runs from the in-session history.
- **Anomaly check** — upload a time-series CSV, pick cyclical / non-cyclical +
  model id, and see the per-cycle/per-window anomaly scores with an adjustable
  threshold and flagged count (`/anomaly/score`).
- **Train model** — train an anomaly bundle from uploaded CSV(s)
  (`/anomaly/train`).

## Prerequisites

- Node 18+ (tested on Node 22).
- The orchestrator running and reachable (default `http://localhost:8000`).
  From `PCIL_dev/`:
  ```
  uvicorn pcil.orchestrator:app --host 0.0.0.0 --port 8000
  ```
  The orchestrator already enables CORS, so the browser app can call it directly.

## Run (development)

```
cd dashboard
npm install
npm run dev
```

Open the URL Vite prints (default `http://localhost:5173`), pick a machine,
and click **Run diagnosis**.

If the orchestrator is on a different host/port (e.g. the NUC), point the
dashboard at it:

```
cp .env.example .env
# edit .env -> VITE_ORCHESTRATOR_URL=http://<nuc-ip>:8000
```

## Build (production)

```
npm run build      # outputs static files to dist/
npm run preview    # serve the built files locally to check
```

`dist/` is a folder of static files — serve it with any static host, or
behind the same reverse proxy as the orchestrator.

## How it maps to the API

| UI element            | Response field                                  |
|-----------------------|-------------------------------------------------|
| KPI cards             | `target_summary` (mean of each target)          |
| Recommendation        | `operator_recommendation` (Gemini text)         |
| "What is driving it"  | `impacts.context[].ranked_feature_impacts`      |
| Evidence              | `recovery_records`                              |
| Header meta           | `impacts.system`, `impacts.context_window`      |

This is a separate, decoupled client (like `rag_frontend/`) — it is not part
of the orchestrator Docker image.

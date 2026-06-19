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

Open the URL Vite prints (default `http://localhost:5173`), pick a system,
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

`dist/` is a folder of static files. The orchestrator's FastAPI app
serves them directly at `/dashboard/` whenever `dashboard/dist/`
exists at `PCIL_PROJECT_ROOT`, or whenever the `DASHBOARD_DIST_DIR`
environment variable points at a valid build. The production
Dockerfile is a two-stage build that runs `npm run build` in a Node
stage and copies the resulting `dist/` into the Python runtime image,
so the NUC operator opens `http://<nuc-ip>:8000/dashboard/` directly
— no Node, npm, or separate static host is required at runtime.

When the dashboard is served from the same origin as the API (i.e.
`/dashboard/` on port 8000), the bundled client uses same-origin
requests by default. Override `VITE_ORCHESTRATOR_URL` only if you are
running the dashboard from a different host than the orchestrator,
e.g. `npm run dev` against a remote NUC.

## How it maps to the API

| UI element            | Response field                                  |
|-----------------------|-------------------------------------------------|
| KPI cards             | `target_summary` (mean of each target)          |
| Recommendation        | `operator_recommendation` (Gemini text)         |
| "What is driving it"  | `impacts.context[].ranked_feature_impacts`      |
| Evidence              | `recovery_records`                              |
| Header meta           | `impacts.system`, `impacts.context_window`      |

This is a decoupled React/Vite client — the build (`dist/`) is bundled
into the orchestrator Docker image and served at `/dashboard/` by the
same FastAPI process. Source still lives outside `pcil/` so the dev
workflow stays Node-only.

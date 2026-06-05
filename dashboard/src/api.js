// Thin client for the PCIL Job Orchestrator. The dashboard holds NO pipeline
// logic — it just calls the API and renders the JSON.

// Resolution order:
//   1. VITE_ORCHESTRATOR_URL (vite dev server, or any cross-origin deploy).
//   2. Empty string → same-origin requests. This is the production path:
//      the FastAPI app serves the dashboard at /dashboard from its own
//      origin, so /pipeline/run etc. resolve to that same host:port.
const BASE = import.meta.env.VITE_ORCHESTRATOR_URL || ''

// Human-readable label for the API pill in the header.
export const ORCHESTRATOR_URL =
  BASE || (typeof window !== 'undefined' ? window.location.origin : 'same-origin')

async function handle(res) {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      detail = body.detail || JSON.stringify(body)
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return res.json()
}

// ── Diagnosis (Pipeline #1-#3) ─────────────────────────────────────

// Run on the CSV named in the config's trigger.source.
export function runDiagnosis(configPath) {
  return fetch(`${BASE}/pipeline/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config_path: configPath, persist: false }),
  }).then(handle)
}

// Run on an uploaded shop-floor CSV (multipart). Same response shape as
// runDiagnosis, so the same components render it.
export function runDiagnosisUpload(configPath, file) {
  const fd = new FormData()
  fd.append('config_path', configPath)
  fd.append('file', file)
  return fetch(`${BASE}/pipeline/run_csv`, { method: 'POST', body: fd }).then(handle)
}

// Pull the configured slice and write it to disk as context_window_*.csv.
export function saveCsv(configPath) {
  return fetch(`${BASE}/pipeline/save_csv`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config_path: configPath }),
  }).then(handle)
}

// ── Anomaly detection ──────────────────────────────────────────────

// Score time-series rows. /anomaly/score takes JSON rows (not a file), so the
// caller parses the CSV client-side first.
export function scoreAnomaly({ rows, modelType, modelId }) {
  return fetch(`${BASE}/anomaly/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      data: rows,
      model_type: modelType,
      model_id: modelId || null,
    }),
  }).then(handle)
}

// Train an anomaly bundle from uploaded CSV(s) (multipart).
export function trainAnomaly(f) {
  const fd = new FormData()
  fd.append('model_type', f.modelType)
  fd.append('training_mode', f.trainingMode)
  if (f.modelId) fd.append('model_id', f.modelId)
  if (f.modelName) fd.append('model_name', f.modelName)
  if (f.file) fd.append('file', f.file)
  if (f.cleanFile) fd.append('clean_file', f.cleanFile)
  if (f.anomalyFile) fd.append('anomaly_file', f.anomalyFile)
  return fetch(`${BASE}/anomaly/train`, { method: 'POST', body: fd }).then(handle)
}

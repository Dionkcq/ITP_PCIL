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

// Service metadata — used by the header's connectivity indicator.
export function ping() {
  return fetch(`${BASE}/`).then(handle)
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

// ── Config recipes (Config tab editor) ─────────────────────────────

// List the recipes under systems/.
export function listConfigs() {
  return fetch(`${BASE}/configs`).then(handle)
}

// Load one recipe as structured data (never raw YAML).
export function loadConfigRecipe(path) {
  return fetch(`${BASE}/configs/load?path=${encodeURIComponent(path)}`).then(handle)
}

// Dry-run validation: same checks as save, nothing written.
export function validateConfigRecipe({ path, config }) {
  return fetch(`${BASE}/configs/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, config }),
  }).then(handle)
}

// Validate + persist. Responds {status:'ok'|'invalid', errors, warnings};
// 'invalid' means nothing was written. Overwrites store a backup server-side.
export function saveConfigRecipe({ path, config, saveAs }) {
  return fetch(`${BASE}/configs/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, config, save_as: saveAs || null }),
  }).then(handle)
}

// Create a brand-new system folder + recipe. 409 if it already exists.
export function createConfigRecipe({ system, name, config }) {
  return fetch(`${BASE}/configs/create`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ system, name: name || 'config', config }),
  }).then(handle)
}

// Delete a recipe — server moves it into .backups/ (recoverable from disk).
export function deleteConfigRecipe(path) {
  return fetch(`${BASE}/configs/delete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  }).then(handle)
}

// ── Anomaly detection ──────────────────────────────────────────────

// List the trained .pkl bundles in data/ — drives the bundle indicator.
export function listAnomalyModels() {
  return fetch(`${BASE}/anomaly/models`).then(handle)
}

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
  // Irregular-only knobs; the endpoint has safe defaults for both.
  if (f.valueColumn) fd.append('value_column', f.valueColumn)
  if (f.windowSeconds) fd.append('window_seconds', f.windowSeconds)
  return fetch(`${BASE}/anomaly/train`, { method: 'POST', body: fd }).then(handle)
}

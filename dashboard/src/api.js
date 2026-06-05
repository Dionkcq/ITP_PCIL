// Thin client for the PCIL Job Orchestrator. The dashboard holds NO pipeline
// logic — it just calls the API and renders the JSON.

const BASE =
  import.meta.env.VITE_ORCHESTRATOR_URL || 'http://localhost:8000'

export const ORCHESTRATOR_URL = BASE

export async function runDiagnosis(configPath) {
  const res = await fetch(`${BASE}/pipeline/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config_path: configPath, persist: false }),
  })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return res.json()
}

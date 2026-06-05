import { useState } from 'react'
import { runDiagnosis, ORCHESTRATOR_URL } from './api.js'
import { KpiCards, Recommendation, ImpactBars, EvidenceList } from './components.jsx'

// Add more recipes here as other machines get configs.
const CONFIGS = [
  { label: 'Inkjet Printer', value: 'machines/inkjet_printer/config.yaml' },
]

export default function App() {
  const [config, setConfig] = useState(CONFIGS[0].value)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [data, setData] = useState(null)

  async function handleRun() {
    setLoading(true)
    setError(null)
    try {
      const result = await runDiagnosis(config)
      setData(result)
    } catch (e) {
      setError(e.message)
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">PCIL</span>
          <span className="subtitle">Operator Dashboard</span>
        </div>
        <div className="api-pill">API: {ORCHESTRATOR_URL}</div>
      </header>

      <div className="controls">
        <label className="field">
          <span>Machine</span>
          <select value={config} onChange={(e) => setConfig(e.target.value)}>
            {CONFIGS.map((c) => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>
        </label>
        <button className="run-btn" onClick={handleRun} disabled={loading}>
          {loading ? 'Running diagnosis…' : 'Run diagnosis'}
        </button>
      </div>

      {error && <div className="banner error">Error: {error}</div>}
      {!data && !error && (
        <div className="placeholder">
          Select a machine and run a diagnosis to see the operator recommendation,
          the KPIs for the window, what is driving them, and the supporting evidence.
        </div>
      )}

      {data && (
        <main className="results">
          <MetaBar data={data} />
          <KpiCards summary={data.target_summary} />
          <Recommendation text={data.operator_recommendation} />
          <ImpactBars impacts={data.impacts} />
          <EvidenceList records={data.recovery_records} />
        </main>
      )}
    </div>
  )
}

function MetaBar({ data }) {
  const system = data.impacts?.system ?? 'unknown'
  const cw = data.impacts?.context_window ?? {}
  const from = cw.time_from ?? cw.start ?? '—'
  const to = cw.time_to ?? cw.end ?? '—'
  return (
    <div className="meta">
      <span><strong>System</strong> {system}</span>
      <span><strong>Window</strong> {from} &rarr; {to}</span>
      <span><strong>Rows</strong> {data.input_rows}</span>
    </div>
  )
}

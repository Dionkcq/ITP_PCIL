import { useEffect, useState } from 'react'
import { listConfigs, runDiagnosis, runDiagnosisUpload, saveCsv } from '../api.js'
import { DiagnosisResult } from '../components.jsx'

// Fallback while /configs hasn't answered (or on older orchestrators).
const CONFIGS = [
  { label: 'Inkjet Printer', value: 'machines/inkjet_printer/config.yaml' },
]

export default function DiagnosisTab() {
  const [configs, setConfigs] = useState(CONFIGS)
  const [config, setConfig] = useState(CONFIGS[0].value)

  // Recipes created in the Config tab (save-as) should be selectable
  // here, so the list comes from the API instead of being hardcoded.
  useEffect(() => {
    listConfigs()
      .then((r) => {
        if (r.configs?.length) {
          setConfigs(
            r.configs.map((c) => ({
              label: `${c.machine} — ${c.name}`,
              value: c.config_path,
            })),
          )
        }
      })
      .catch(() => {}) // keep the fallback list
  }, [])
  const [mode, setMode] = useState('configured') // 'configured' | 'upload'
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [data, setData] = useState(null)
  const [history, setHistory] = useState([])
  const [exportMsg, setExportMsg] = useState(null)

  async function handleRun() {
    setLoading(true)
    setError(null)
    setExportMsg(null)
    try {
      let result
      if (mode === 'upload') {
        if (!file) throw new Error('Choose a CSV file to upload first.')
        result = await runDiagnosisUpload(config, file)
      } else {
        result = await runDiagnosis(config)
      }
      setData(result)
      setHistory((h) =>
        [
          {
            at: new Date().toLocaleTimeString(),
            name: mode === 'upload' ? file.name : 'configured source',
            data: result,
          },
          ...h,
        ].slice(0, 10),
      )
    } catch (e) {
      setError(e.message)
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  function downloadJson() {
    if (!data) return
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'pcil_diagnosis.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  async function handleExport() {
    setExportMsg(null)
    try {
      const r = await saveCsv(config)
      setExportMsg(`Saved ${r.rows} rows -> ${r.path}`)
    } catch (e) {
      setExportMsg(`Export failed: ${e.message}`)
    }
  }

  return (
    <div className="tabpane">
      <section className="controls-card">
        <div className="seg">
          <button
            className={mode === 'configured' ? 'on' : ''}
            onClick={() => setMode('configured')}
          >
            Run on configured source
          </button>
          <button
            className={mode === 'upload' ? 'on' : ''}
            onClick={() => setMode('upload')}
          >
            Upload a CSV
          </button>
        </div>

        <label className="field">
          <span>Machine / recipe</span>
          <select value={config} onChange={(e) => setConfig(e.target.value)}>
            {configs.map((c) => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>
        </label>

        {mode === 'upload' && (
          <label className="field">
            <span>Shop-floor CSV</span>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files[0] ?? null)}
            />
          </label>
        )}

        <button className="run-btn" onClick={handleRun} disabled={loading}>
          {loading ? 'Running diagnosis…' : 'Run diagnosis'}
        </button>
      </section>

      {error && <div className="banner error">Error: {error}</div>}
      {!data && !error && (
        <div className="placeholder">
          Run a diagnosis to see the operator recommendation, the KPIs for the
          window, what is driving them, and the supporting evidence.
        </div>
      )}

      {data && (
        <main className="results">
          <div className="actions no-print">
            <button onClick={downloadJson}>Download JSON</button>
            <button onClick={() => window.print()}>Print / PDF</button>
            {mode === 'configured' && (
              <button onClick={handleExport}>Export context-window CSV</button>
            )}
            {exportMsg && <span className="export-msg">{exportMsg}</span>}
          </div>
          <DiagnosisResult data={data} />
        </main>
      )}

      {history.length > 0 && (
        <section className="history no-print">
          <h2>History</h2>
          {history.map((h, i) => (
            <button
              key={i}
              className="history-item"
              onClick={() => setData(h.data)}
            >
              <span className="h-time">{h.at}</span>
              <span className="h-name">{h.name}</span>
              <span className="h-oee">
                OEE{' '}
                {h.data.target_summary?.oee != null
                  ? `${(h.data.target_summary.oee * 100).toFixed(1)}%`
                  : '—'}
              </span>
            </button>
          ))}
        </section>
      )}
    </div>
  )
}

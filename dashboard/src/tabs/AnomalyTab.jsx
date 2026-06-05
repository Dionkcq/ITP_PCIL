import { useState } from 'react'
import { scoreAnomaly } from '../api.js'
import { parseCsvFile } from '../csv.js'
import { ScoreStrip } from '../components.jsx'

function percentile(arr, p) {
  if (!arr.length) return 0
  const s = [...arr].sort((a, b) => a - b)
  const idx = Math.min(s.length - 1, Math.floor((p / 100) * s.length))
  return s[idx]
}

export default function AnomalyTab() {
  const [file, setFile] = useState(null)
  const [modelType, setModelType] = useState('cyclical')
  const [modelId, setModelId] = useState('inkjet_01')
  const [skipRows, setSkipRows] = useState(0)
  const [pctl, setPctl] = useState(90)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  async function handleScore() {
    setLoading(true)
    setError(null)
    try {
      if (!file) throw new Error('Choose a CSV file first.')
      const rows = await parseCsvFile(file, { skipRows: Number(skipRows) || 0 })
      if (!rows.length) {
        throw new Error('No rows parsed from the CSV (check the skip-rows value).')
      }
      const res = await scoreAnomaly({ rows, modelType, modelId })
      setResult(res)
    } catch (e) {
      setError(e.message)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  const scores = result?.anomaly_scores ?? []
  const hasScores = scores.length > 0
  const threshold = hasScores ? percentile(scores, pctl) : null
  const flagged = threshold != null ? scores.filter((s) => s > threshold).length : 0
  const mean = hasScores ? scores.reduce((a, b) => a + b, 0) / scores.length : 0

  return (
    <div className="tabpane">
      <section className="controls-card col">
        <div className="row">
          <label className="field">
            <span>Time-series CSV</span>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files[0] ?? null)}
            />
          </label>
          <label className="field">
            <span>Model type</span>
            <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
              <option value="cyclical">cyclical</option>
              <option value="non_cyclical">non_cyclical</option>
            </select>
          </label>
          <label className="field">
            <span>Model ID</span>
            <input value={modelId} onChange={(e) => setModelId(e.target.value)} />
          </label>
          <label className="field">
            <span>Skip header rows</span>
            <input
              type="number"
              min="0"
              value={skipRows}
              onChange={(e) => setSkipRows(e.target.value)}
            />
          </label>
        </div>
        <div className="hint">
          cyclical expects <code>timestamp, signal_value, machine_id</code>;
          non_cyclical expects the channel columns the bundle was trained on. The
          raw acoustic recordings have 5 metadata rows — set skip to 5 for those.
        </div>
        <button className="run-btn" onClick={handleScore} disabled={loading}>
          {loading ? 'Scoring…' : 'Score anomalies'}
        </button>
      </section>

      {error && <div className="banner error">Error: {error}</div>}

      {result && (
        <main className="results">
          <div className="meta">
            <span><strong>Model</strong> {result.model_type} / {result.model_id}</span>
            <span><strong>Input rows</strong> {result.input_rows}</span>
            <span>
              <strong>{result.model_type === 'cyclical' ? 'Cycles' : 'Windows'} scored</strong>{' '}
              {result.cycles_scored ?? result.windows_scored}
            </span>
          </div>

          <section>
            <div className="section-head">
              <h2>Anomaly scores</h2>
              <label className="thresh">
                flag above p
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={pctl}
                  onChange={(e) => setPctl(Number(e.target.value))}
                />
                {' = '}
                {threshold != null ? threshold.toFixed(3) : '—'}
              </label>
            </div>
            <ScoreStrip scores={scores} threshold={threshold} />
            {hasScores && (
              <div className="stat-row">
                <span>flagged <strong>{flagged}</strong> / {scores.length}</span>
                <span>min {Math.min(...scores).toFixed(3)}</span>
                <span>mean {mean.toFixed(3)}</span>
                <span>max {Math.max(...scores).toFixed(3)}</span>
              </div>
            )}
          </section>
        </main>
      )}
    </div>
  )
}

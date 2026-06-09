import { useState } from 'react'
import { scoreAnomaly } from '../api.js'
import { parseCsvFile } from '../csv.js'
import { ScoreChart } from '../components.jsx'

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
  // 'server' = the 95th-percentile-of-training threshold stored in the
  // bundle at train time; 'percentile' = computed client-side over THIS
  // result's scores. Server is the default when the bundle provides one.
  const [threshMode, setThreshMode] = useState('server')
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
  const serverThreshold = result?.threshold ?? null
  const mode = serverThreshold != null ? threshMode : 'percentile'
  const threshold = hasScores
    ? mode === 'server'
      ? serverThreshold
      : percentile(scores, pctl)
    : null
  const flagged = threshold != null ? scores.filter((s) => s > threshold).length : 0
  const mean = hasScores ? scores.reduce((a, b) => a + b, 0) / scores.length : 0
  const labels =
    result?.cycle_start_timestamps ?? result?.window_start_timestamps ?? null

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
              <option value="irregular">irregular</option>
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
          non_cyclical expects the channel columns the bundle was trained on;
          irregular expects <code>timestamp, machine_id</code> (+ the value column
          if the bundle was trained with one). The raw acoustic recordings have
          5 metadata rows — set skip to 5 for those.
        </div>
        <button className="run-btn" onClick={handleScore} disabled={loading}>
          {loading ? 'Scoring…' : 'Score anomalies'}
        </button>
      </section>

      {error && <div className="banner error">Error: {error}</div>}
      {!result && !error && (
        <div className="placeholder">
          Upload a time-series CSV and score it to see per-cycle / per-window
          anomaly scores charted against the model&apos;s threshold.
        </div>
      )}

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
              <div className="thresh">
                {serverThreshold != null && (
                  <span className="seg seg-sm">
                    <button
                      className={mode === 'server' ? 'on' : ''}
                      onClick={() => setThreshMode('server')}
                      title="Threshold stored in the bundle at train time (95th percentile of training scores)"
                    >
                      bundle
                    </button>
                    <button
                      className={mode === 'percentile' ? 'on' : ''}
                      onClick={() => setThreshMode('percentile')}
                      title="Percentile computed over this result's scores"
                    >
                      percentile
                    </button>
                  </span>
                )}
                {mode === 'percentile' && (
                  <>
                    p
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={pctl}
                      onChange={(e) => setPctl(Number(e.target.value))}
                    />
                  </>
                )}
                {' = '}
                {threshold != null ? threshold.toFixed(3) : '—'}
              </div>
            </div>
            <ScoreChart scores={scores} threshold={threshold} labels={labels} />
            {hasScores && (
              <div className="stat-row">
                <span>
                  flagged <strong className={flagged > 0 ? 'flag-count' : ''}>{flagged}</strong>{' '}
                  / {scores.length}
                </span>
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

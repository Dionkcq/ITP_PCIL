import { useState } from 'react'
import { trainAnomaly } from '../api.js'

export default function TrainTab() {
  const [modelType, setModelType] = useState('cyclical')
  const [modelId, setModelId] = useState('inkjet_01')
  const [modelName, setModelName] = useState('autoencoder')
  const [file, setFile] = useState(null)
  const [cleanFile, setCleanFile] = useState(null)
  const [anomalyFile, setAnomalyFile] = useState(null)
  const [valueColumn, setValueColumn] = useState('')
  const [windowSeconds, setWindowSeconds] = useState('1.0')
  // Rows to skip before the CSV header. Raw WebDAQ acoustic exports
  // (non_cyclical) carry a 5-line device-info preamble; clean CSVs start at 0.
  const [headerSkiprows, setHeaderSkiprows] = useState('0')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  // cyclical + irregular train on one normal-operation CSV;
  // non_cyclical trains on a labelled clean/anomaly pair.
  const singleFile = modelType === 'cyclical' || modelType === 'irregular'

  async function handleTrain() {
    setLoading(true)
    setError(null)
    try {
      const trainingMode = singleFile ? 'normal_only' : 'clean_vs_anomaly'
      if (singleFile && !file) {
        throw new Error('Training needs a normal-operation CSV.')
      }
      if (modelType === 'non_cyclical' && (!cleanFile || !anomalyFile)) {
        throw new Error('Non-cyclical training needs both a clean and an anomaly CSV.')
      }
      const res = await trainAnomaly({
        modelType,
        trainingMode,
        modelId,
        modelName: modelType === 'cyclical' ? modelName : null,
        file,
        cleanFile,
        anomalyFile,
        valueColumn: modelType === 'irregular' ? valueColumn : null,
        windowSeconds: modelType === 'irregular' ? windowSeconds : null,
        headerSkiprows,
      })
      setResult(res)
    } catch (e) {
      setError(e.message)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="tabpane">
      <section className="controls-card col">
        <div className="row">
          <label className="field">
            <span>Model type</span>
            <select
              value={modelType}
              onChange={(e) => {
                const t = e.target.value
                setModelType(t)
                // Raw WebDAQ acoustic files (non_cyclical) need 5; others 0.
                setHeaderSkiprows(t === 'non_cyclical' ? '5' : '0')
              }}
            >
              <option value="cyclical">cyclical</option>
              <option value="non_cyclical">non_cyclical</option>
              <option value="irregular">irregular</option>
            </select>
          </label>
          <label className="field">
            <span>Model ID</span>
            <input value={modelId} onChange={(e) => setModelId(e.target.value)} />
          </label>
          {modelType === 'cyclical' && (
            <label className="field">
              <span>Algorithm</span>
              <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
                <option value="autoencoder">autoencoder (1D CNN)</option>
                <option value="isolation_forest">isolation_forest</option>
              </select>
            </label>
          )}
          {modelType === 'irregular' && (
            <>
              <label className="field">
                <span>Value column (optional)</span>
                <input
                  value={valueColumn}
                  placeholder="e.g. signal_value"
                  onChange={(e) => setValueColumn(e.target.value)}
                />
              </label>
              <label className="field">
                <span>Window (seconds)</span>
                <input
                  value={windowSeconds}
                  onChange={(e) => setWindowSeconds(e.target.value)}
                />
              </label>
            </>
          )}
        </div>

        <label className="field">
          <span>Header rows to skip</span>
          <input
            type="number"
            min="0"
            value={headerSkiprows}
            onChange={(e) => setHeaderSkiprows(e.target.value)}
          />
          <span className="hint">
            5 for raw WebDAQ acoustic exports (skips the device-info preamble
            before the header); 0 for already-clean CSVs.
          </span>
        </label>

        {singleFile ? (
          <label className="field">
            <span>
              {modelType === 'cyclical'
                ? 'Normal-data CSV (signal_value / machine_id / timestamp)'
                : 'Normal-data CSV (machine_id / timestamp, + value column if set)'}
            </span>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files[0] ?? null)}
            />
          </label>
        ) : (
          <div className="row">
            <label className="field">
              <span>Clean recording CSV</span>
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setCleanFile(e.target.files[0] ?? null)}
              />
            </label>
            <label className="field">
              <span>Anomaly recording CSV</span>
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setAnomalyFile(e.target.files[0] ?? null)}
              />
            </label>
          </div>
        )}

        <div className="hint">
          Training writes a bundle to <code>data/&lt;model_type&gt;_&lt;model_id&gt;.pkl</code>,
          which <code>/anomaly/score</code> then loads. The 1D CNN can take a little while.
        </div>
        <button className="run-btn" onClick={handleTrain} disabled={loading}>
          {loading ? 'Training…' : 'Train model'}
        </button>
      </section>

      {error && <div className="banner error">Error: {error}</div>}
      {result && (
        <section className="results train-result">
          <h2>Training complete</h2>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </section>
      )}
    </div>
  )
}

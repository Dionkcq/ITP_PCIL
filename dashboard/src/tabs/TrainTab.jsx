import { useState } from 'react'
import { trainAnomaly } from '../api.js'

export default function TrainTab() {
  const [modelType, setModelType] = useState('cyclical')
  const [modelId, setModelId] = useState('inkjet_01')
  const [modelName, setModelName] = useState('autoencoder')
  const [file, setFile] = useState(null)
  const [cleanFile, setCleanFile] = useState(null)
  const [anomalyFile, setAnomalyFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  async function handleTrain() {
    setLoading(true)
    setError(null)
    try {
      const trainingMode =
        modelType === 'cyclical' ? 'normal_only' : 'clean_vs_anomaly'
      if (modelType === 'cyclical' && !file) {
        throw new Error('Cyclical training needs a normal-data CSV.')
      }
      if (modelType === 'non_cyclical' && (!cleanFile || !anomalyFile)) {
        throw new Error('Non-cyclical training needs both a clean and an anomaly CSV.')
      }
      const res = await trainAnomaly({
        modelType,
        trainingMode,
        modelId,
        modelName,
        file,
        cleanFile,
        anomalyFile,
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
            <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
              <option value="cyclical">cyclical</option>
              <option value="non_cyclical">non_cyclical</option>
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
        </div>

        {modelType === 'cyclical' ? (
          <label className="field">
            <span>Normal-data CSV (signal_value / machine_id / timestamp)</span>
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

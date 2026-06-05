import { useState } from 'react'

// ── Header meta ────────────────────────────────────────────────────
export function MetaBar({ data }) {
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

// ── KPI cards ──────────────────────────────────────────────────────
const TARGET_ORDER = ['availability', 'performance', 'quality', 'oee']
const TARGET_LABELS = {
  availability: 'Availability',
  performance: 'Performance',
  quality: 'Quality',
  oee: 'OEE',
}

const pct = (v) => `${(v * 100).toFixed(1)}%`
const tone = (v) => (v >= 0.85 ? 'good' : v >= 0.6 ? 'warn' : 'bad')

export function KpiCards({ summary }) {
  if (!summary || Object.keys(summary).length === 0) return null
  const ordered = [
    ...TARGET_ORDER.filter((k) => k in summary),
    ...Object.keys(summary).filter((k) => !TARGET_ORDER.includes(k)),
  ]
  return (
    <section className="kpi-row">
      {ordered.map((k) => (
        <div key={k} className={`kpi-card ${tone(summary[k])}`}>
          <div className="kpi-value">{pct(summary[k])}</div>
          <div className="kpi-label">{TARGET_LABELS[k] ?? k}</div>
        </div>
      ))}
    </section>
  )
}

// ── Operator recommendation ────────────────────────────────────────
const FALLBACK_HINTS = ['not found', 'failed', 'not set', 'No matching recovery']

export function Recommendation({ text }) {
  const isFallback = FALLBACK_HINTS.some((h) => (text || '').includes(h))
  return (
    <section className={`reco ${isFallback ? 'reco-fallback' : ''}`}>
      <h2>Recommendation</h2>
      <p>{text || 'No recommendation returned.'}</p>
      {isFallback && (
        <div className="reco-note">
          This is a fallback message, not a live LLM result. Check that
          GEMINI_API_KEY is set and data/RAG/ is mounted.
        </div>
      )}
    </section>
  )
}

// ── Ranked feature impacts ─────────────────────────────────────────
export function ImpactBars({ impacts }) {
  const blocks = impacts?.context ?? []
  const initial =
    blocks.find((b) => b.target === 'oee')?.target ?? blocks[0]?.target ?? ''
  const [target, setTarget] = useState(initial)

  const block = blocks.find((b) => b.target === target)
  const feats = block?.ranked_feature_impacts ?? []
  const maxAbs = Math.max(
    1e-9,
    ...feats.map((f) => Math.abs(f.standardized_impact_score)),
  )

  return (
    <section className="impacts">
      <div className="section-head">
        <h2>What is driving it</h2>
        {blocks.length > 0 && (
          <select value={target} onChange={(e) => setTarget(e.target.value)}>
            {blocks.map((b) => (
              <option key={b.target} value={b.target}>{b.target}</option>
            ))}
          </select>
        )}
      </div>
      <div className="bars">
        {feats.map((f) => {
          const v = f.standardized_impact_score
          const width = (Math.abs(v) / maxAbs) * 100
          return (
            <div key={f.feature} className="bar-row" title={f.description || ''}>
              <div className="bar-label">{f.feature}</div>
              <div className="bar-track">
                <div
                  className={`bar-fill ${v < 0 ? 'neg' : 'pos'}`}
                  style={{ width: `${width}%` }}
                />
              </div>
              <div className="bar-val">{v >= 0 ? '+' : ''}{v.toFixed(2)}</div>
            </div>
          )
        })}
        {feats.length === 0 && (
          <div className="muted">No ranked impacts for this target.</div>
        )}
      </div>
    </section>
  )
}

// ── Evidence (retrieved recovery records) ──────────────────────────
export function EvidenceList({ records }) {
  const recs = records ?? []
  return (
    <section className="evidence">
      <h2>Evidence ({recs.length})</h2>
      {recs.length === 0 && (
        <div className="muted">No recovery records retrieved for this window.</div>
      )}
      {recs.map((r, i) => (
        <details key={i} className="evidence-item">
          <summary>
            <span className="src">{r.source_doc}</span>
            {r.error}
          </summary>
          <div className="evidence-body">
            <p><strong>Cause:</strong> {r.cause}</p>
            <p><strong>Recovery:</strong> {r.recovery}</p>
          </div>
        </details>
      ))}
    </section>
  )
}

// ── Full diagnosis (reused for the current run and history detail) ─
export function DiagnosisResult({ data }) {
  return (
    <>
      <MetaBar data={data} />
      <KpiCards summary={data.target_summary} />
      <Recommendation text={data.operator_recommendation} />
      <ImpactBars impacts={data.impacts} />
      <EvidenceList records={data.recovery_records} />
    </>
  )
}

// ── Anomaly score strip ────────────────────────────────────────────
export function ScoreStrip({ scores, threshold }) {
  if (!scores || scores.length === 0) {
    return <div className="muted">No cycles / windows detected in the input.</div>
  }
  const max = Math.max(...scores, 1e-9)
  return (
    <div className="score-strip">
      {scores.map((s, i) => {
        const h = Math.max(2, (s / max) * 100)
        const flagged = threshold != null && s > threshold
        return (
          <div
            key={i}
            className={`score-bar ${flagged ? 'flagged' : ''}`}
            style={{ height: `${h}%` }}
            title={`#${i}: ${s.toFixed(3)}${flagged ? ' (flagged)' : ''}`}
          />
        )
      })}
    </div>
  )
}

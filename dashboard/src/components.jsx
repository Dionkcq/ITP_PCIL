import { useState } from 'react'

// ── Header meta ────────────────────────────────────────────────────

// ISO timestamps are precise but noisy in a meta bar — show
// "2026-05-15 09:00:30" instead of "2026-05-15T09:00:30+00:00".
function shortTime(iso) {
  if (typeof iso !== 'string') return iso
  return iso.replace('T', ' ').replace(/(\+00:00|Z)$/, '')
}

export function MetaBar({ data }) {
  const system = data.impacts?.system ?? 'unknown'
  const cw = data.impacts?.context_window ?? {}
  // The Week-3 impacts schema names these start_time / end_time; the
  // older spellings are kept as fallbacks for archived responses.
  const from = cw.start_time ?? cw.time_from ?? cw.start ?? '—'
  const to = cw.end_time ?? cw.time_to ?? cw.end ?? '—'
  return (
    <div className="meta">
      <span><strong>System</strong> {system}</span>
      <span><strong>Window</strong> {shortTime(from)} &rarr; {shortTime(to)}</span>
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

// Small SVG ring gauge — fill fraction matches the KPI value.
function Ring({ value }) {
  const r = 23
  const c = 2 * Math.PI * r
  const frac = Math.max(0, Math.min(1, value))
  return (
    <svg className="ring" viewBox="0 0 56 56" width="56" height="56" aria-hidden="true">
      <circle className="ring-track" cx="28" cy="28" r={r} />
      <circle
        className="ring-fill"
        cx="28"
        cy="28"
        r={r}
        strokeDasharray={`${frac * c} ${c}`}
        transform="rotate(-90 28 28)"
      />
    </svg>
  )
}

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
          <div className="kpi-body">
            <div className="kpi-value">{pct(summary[k])}</div>
            <div className="kpi-label">{TARGET_LABELS[k] ?? k}</div>
          </div>
          <Ring value={summary[k]} />
        </div>
      ))}
    </section>
  )
}

// ── Operator recommendation ────────────────────────────────────────
const FALLBACK_HINTS = ['not found', 'failed', 'not set', 'No matching recovery']

// Status-specific guidance so the note points at the ACTUAL failure (e.g. an
// LLM-key problem must not blame the RAG store, which may be perfectly fine).
const FALLBACK_NOTE = {
  llm_unavailable:
    'Records were retrieved, but the LLM step failed — check GEMINI_API_KEY and that the container can reach the network.',
  rag_unavailable:
    'The RAG store is unavailable — in file mode mount data/RAG/; in postgres mode check the RAG store / DATABASE_URL.',
  retrieval_failed:
    'Retrieval errored before the LLM ran — see the message above for the cause.',
  no_records:
    'No recovery records matched this window, so there was nothing to ground a recommendation on.',
}

export function Recommendation({ text, status }) {
  // Canonical signal is the structured recommendation_status from the API
  // (ok | no_records | retrieval_failed | llm_unavailable | rag_unavailable).
  // The legacy string sniff is kept only for archived/exported runs saved
  // before recommendation_status existed (status is undefined for those).
  const isFallback =
    status != null
      ? status !== 'ok'
      : FALLBACK_HINTS.some((h) => (text || '').includes(h))
  const label = status ? status.replaceAll('_', ' ') : 'legacy response'
  return (
    <section className={`reco ${isFallback ? 'reco-fallback' : ''}`}>
      <h2>Recommendation <span className="source-pill">{label}</span></h2>
      <p>{text || 'No recommendation returned.'}</p>
      {isFallback && (
        <div className="reco-note">
          {FALLBACK_NOTE[status] ||
            'This is not a live grounded Gemini recommendation.'}
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
  // Bar = live contribution (current normalised value x model weight) when
  // present; fall back to the coefficient share for archived runs saved before
  // contribution existed. ranked_feature_impacts already arrives ordered by it.
  const valOf = (f) =>
    f.standardized_contribution ?? f.standardized_impact_score ?? 0
  const maxAbs = Math.max(1e-9, ...feats.map((f) => Math.abs(valOf(f))))

  return (
    <section className="impacts">
      <div className="section-head">
        <h2>Correlated signals in this window</h2>
        {blocks.length > 0 && (
          <select value={target} onChange={(e) => setTarget(e.target.value)}>
            {blocks.map((b) => (
              <option key={b.target} value={b.target}>{b.target}</option>
            ))}
          </select>
        )}
      </div>
      <p className="muted">
        Ranked by live contribution (current value &times; model weight)
      </p>
      <div className="bars">
        {feats.map((f) => {
          const v = valOf(f)
          const width = (Math.abs(v) / maxAbs) * 100
          const tip = [
            f.description || '',
            f.feature_value != null ? `value ${f.feature_value.toFixed(2)}` : '',
            f.raw_impact_score != null ? `weight ${f.raw_impact_score.toFixed(3)}` : '',
            f.contribution != null ? `contribution ${f.contribution.toFixed(3)}` : '',
          ].filter(Boolean).join('  •  ')
          return (
            <div key={f.feature} className="bar-row" title={tip}>
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
export function BaselineComparison({ comparison }) {
  if (!comparison || comparison.status !== 'available') return null
  const entries = Object.entries(comparison.features ?? {})
    .filter(([, v]) => v.z_score != null)
    .sort((a, b) => Math.abs(b[1].z_score) - Math.abs(a[1].z_score))
    .slice(0, 4)
  if (entries.length === 0) return null
  return (
    <section className="baseline">
      <h2>Baseline comparison</h2>
      <div className="baseline-grid">
        {entries.map(([name, v]) => (
          <div key={name} className="baseline-item">
            <span>{name}</span>
            <strong>{v.z_score >= 0 ? '+' : ''}{v.z_score.toFixed(2)} z</strong>
            <em>{v.direction.replaceAll('_', ' ')}</em>
          </div>
        ))}
      </div>
    </section>
  )
}

export function DiagnosisResult({ data }) {
  return (
    <>
      <MetaBar data={data} />
      <KpiCards summary={data.target_summary} />
      <Recommendation text={data.operator_recommendation} status={data.recommendation_status} />
      <BaselineComparison comparison={data.baseline_comparison} />
      <ImpactBars impacts={data.impacts} />
      <EvidenceList records={data.recovery_records} />
    </>
  )
}

// ── Anomaly score chart ────────────────────────────────────────────
// Hand-rolled SVG bar chart (project decision: no chart-lib dependency).
// Y axis with ticks, a dashed threshold line, and per-bar tooltips that
// include the cycle/window timestamp when the API provides one.
export function ScoreChart({ scores, threshold, labels }) {
  if (!scores || scores.length === 0) {
    return <div className="muted">No cycles / windows detected in the input.</div>
  }
  const W = 860
  const H = 210
  const PAD_L = 52
  const PAD_R = 12
  const PAD_T = 12
  const PAD_B = 26
  const plotW = W - PAD_L - PAD_R
  const plotH = H - PAD_T - PAD_B

  // Headroom so the tallest bar / threshold line never touches the frame.
  const maxV = Math.max(...scores, threshold ?? 0, 1e-9) * 1.08
  const n = scores.length
  const slot = plotW / n
  const gap = slot > 6 ? 2 : slot > 2 ? 1 : 0
  const barW = Math.max(1, slot - gap)
  const yOf = (v) => PAD_T + plotH - (v / maxV) * plotH

  const ticks = [0, maxV / 2, maxV]
  const fmt = (v) => (maxV >= 100 ? v.toFixed(0) : maxV >= 1 ? v.toFixed(2) : v.toFixed(3))
  const labelOf = (i) => {
    const t = labels?.[i]
    return t ? `${t}` : `#${i}`
  }

  return (
    <div className="chart-wrap">
      <svg className="score-chart" viewBox={`0 0 ${W} ${H}`} role="img">
        {ticks.map((t, i) => (
          <g key={i}>
            <line
              className="grid-line"
              x1={PAD_L}
              y1={yOf(t)}
              x2={W - PAD_R}
              y2={yOf(t)}
            />
            <text className="tick-label" x={PAD_L - 8} y={yOf(t) + 4} textAnchor="end">
              {fmt(t)}
            </text>
          </g>
        ))}

        {scores.map((s, i) => {
          const flagged = threshold != null && s > threshold
          const h = Math.max(2, (s / maxV) * plotH)
          return (
            <rect
              key={i}
              className={`chart-bar ${flagged ? 'flagged' : ''}`}
              x={PAD_L + i * slot + gap / 2}
              y={PAD_T + plotH - h}
              width={barW}
              height={h}
              rx={barW > 3 ? 1.5 : 0}
            >
              <title>
                {`${labelOf(i)}\nscore ${s.toFixed(4)}${flagged ? '  (flagged)' : ''}`}
              </title>
            </rect>
          )
        })}

        {threshold != null && (
          <g>
            <line
              className="threshold-line"
              x1={PAD_L}
              y1={yOf(threshold)}
              x2={W - PAD_R}
              y2={yOf(threshold)}
            />
            <text
              className="threshold-label"
              x={W - PAD_R - 4}
              y={yOf(threshold) - 5}
              textAnchor="end"
            >
              threshold {fmt(threshold)}
            </text>
          </g>
        )}

        <text className="tick-label" x={PAD_L} y={H - 8}>
          {labelOf(0)}
        </text>
        <text className="tick-label" x={W - PAD_R} y={H - 8} textAnchor="end">
          {labelOf(n - 1)}
        </text>
      </svg>
    </div>
  )
}

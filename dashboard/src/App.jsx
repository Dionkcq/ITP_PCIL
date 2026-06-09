import { useEffect, useState } from 'react'
import { ORCHESTRATOR_URL, ping } from './api.js'
import DiagnosisTab from './tabs/DiagnosisTab.jsx'
import AnomalyTab from './tabs/AnomalyTab.jsx'
import TrainTab from './tabs/TrainTab.jsx'

const TABS = [
  { id: 'diagnosis', label: 'Diagnosis' },
  { id: 'anomaly', label: 'Anomaly check' },
  { id: 'train', label: 'Train model' },
]

export default function App() {
  const [tab, setTab] = useState('diagnosis')
  // null = unknown (first check pending), true/false afterwards.
  const [online, setOnline] = useState(null)

  useEffect(() => {
    let alive = true
    const check = () =>
      ping()
        .then(() => alive && setOnline(true))
        .catch(() => alive && setOnline(false))
    check()
    const id = setInterval(check, 30000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  // All tabs stay mounted (toggled with display) so their state — including
  // the diagnosis history — survives switching between them.
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">PCIL</span>
          <span className="subtitle">Operator Dashboard</span>
        </div>
        <div
          className="api-pill"
          title={online === false ? 'Orchestrator unreachable' : 'Orchestrator online'}
        >
          <span
            className={`status-dot ${online ? 'on' : online === false ? 'off' : ''}`}
          />
          API: {ORCHESTRATOR_URL}
        </div>
      </header>

      <nav className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`tab ${tab === t.id ? 'active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <div style={{ display: tab === 'diagnosis' ? 'block' : 'none' }}>
        <DiagnosisTab />
      </div>
      <div style={{ display: tab === 'anomaly' ? 'block' : 'none' }}>
        <AnomalyTab />
      </div>
      <div style={{ display: tab === 'train' ? 'block' : 'none' }}>
        <TrainTab />
      </div>
    </div>
  )
}

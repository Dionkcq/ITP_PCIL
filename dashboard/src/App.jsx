import { useState } from 'react'
import { ORCHESTRATOR_URL } from './api.js'
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

  // All tabs stay mounted (toggled with display) so their state — including
  // the diagnosis history — survives switching between them.
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">PCIL</span>
          <span className="subtitle">Operator Dashboard</span>
        </div>
        <div className="api-pill">API: {ORCHESTRATOR_URL}</div>
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

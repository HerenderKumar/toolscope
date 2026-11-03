import { useState } from 'react'
import Compare from './Compare'
import Failures from './Failures'
import ToolHealth from './ToolHealth'

const VIEWS = { Compare, Failures, 'Tool health': ToolHealth }
const SLUGS = { Compare: 'compare', Failures: 'failures', 'Tool health': 'tools' }

// Deep links: #compare, #failures, #tools or #tool=<tool_id> (opens that tool in Tool health).
function initialView() {
  const h = window.location.hash.slice(1)
  if (h.startsWith('tool=')) return 'Tool health'
  return Object.keys(SLUGS).find((v) => SLUGS[v] === h) || 'Compare'
}

export default function App() {
  const [view, setViewState] = useState(initialView)
  const setView = (v) => {
    window.location.hash = SLUGS[v]
    setViewState(v)
  }
  const View = VIEWS[view]
  return (
    <div className="app">
      <header>
        <div>
          <h1>ToolScope</h1>
          <p className="tagline">Inspect MCP tool discovery, compare retrievers, and measure description-edit effects.</p>
        </div>
        <nav>
          {Object.keys(VIEWS).map((v) => (
            <button key={v} className={v === view ? 'active' : ''} onClick={() => setView(v)}>{v}</button>
          ))}
        </nav>
      </header>
      <main><View /></main>
      <footer className="muted small">LiveMCPBench tools (69 servers, 525 tools), 95 tasks. Read-only: every number comes from results/ and logs/.</footer>
    </div>
  )
}

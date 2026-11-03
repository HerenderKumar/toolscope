import { useEffect, useState } from 'react'
import { api } from './api'
import { useLoad } from './hooks'
import { Status } from './components'

const FLAG_NAMES = {
  empty_description: 'empty',
  short_description: 'short',
  non_english_description: 'non-English',
  undocumented_params: 'param docs missing',
}

function Rank({ pair }) {
  const [before, after] = pair
  const cls = after <= 10 ? 'good' : after < before ? 'better' : after > before ? 'worse' : ''
  return <span className={`rank ${cls}`}>{before} → {after}</span>
}

// "#tool=<id>&rewrite" preloads the stored query-blind rewrite and measures it once (demo / screenshots).
const autoRewrite = () => window.location.hash.includes('&rewrite')

function WhatIfEditor({ tool }) {
  const [text, setText] = useState(
    autoRewrite() && tool.rewrite ? tool.rewrite.rewritten : tool.card.description || '')
  const [tasks, setTasks] = useState(tool.tasks.slice(0, 5).map((t) => t.task_id))
  const [state, setState] = useState({ busy: false, error: null, results: null })
  const run = (description = text) => {
    setState({ busy: true, error: null, results: null })
    api.whatif(tool.card.tool_id, description, tasks)
      .then((r) => setState({ busy: false, error: null, results: r.results }))
      .catch((error) => setState({ busy: false, error, results: null }))
  }
  useEffect(() => {
    if (autoRewrite() && tool.rewrite) run(tool.rewrite.rewritten)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  return (
    <div className="card">
      <h4>What-if editor</h4>
      <p className="note">Edit the description and measure the tool’s rank for the tasks that need it. Computed in memory; nothing is saved.
        The first run loads the embedding model (a few seconds).</p>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={7} />
      <div className="row">
        <button onClick={() => setText(tool.card.description || '')}>Original</button>
        {tool.rewrite && <button onClick={() => setText(tool.rewrite.rewritten)}>Query-blind rewrite (DeepSeek V3.2)</button>}
        <button className="primary" disabled={state.busy || !text.trim() || !tasks.length} onClick={() => run()}>
          {state.busy ? 'Measuring…' : 'Measure rank'}
        </button>
      </div>
      <div className="tasks">
        {tool.tasks.map((t) => (
          <label key={t.task_id} className="small">
            <input type="checkbox" checked={tasks.includes(t.task_id)}
                   onChange={(e) => setTasks(e.target.checked ? [...tasks, t.task_id] : tasks.filter((x) => x !== t.task_id))} />
            {t.question.slice(0, 110)}
          </label>
        ))}
      </div>
      {state.error && <p className="error">{state.error.message}</p>}
      {state.results && (
        <table>
          <thead><tr><th>Task</th><th>BM25 rank</th><th>Semantic rank</th></tr></thead>
          <tbody>
            {state.results.map((r) => (
              <tr key={r.task_id}><td className="small">{r.query.slice(0, 90)}</td><td><Rank pair={r.bm25} /></td><td><Rank pair={r.semantic} /></td></tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

function ToolDetail({ toolId }) {
  const { data, error, loading } = useLoad(() => api.tool(toolId), [toolId])
  if (!data) return <Status loading={loading} error={error} />
  return (
    <div className="detail">
      <h3>{data.card.tool_id}</h3>
      <p className="muted small">{data.card.server_name}</p>
      <div className="card">
        <h4>Current description</h4>
        <p className="desc">{data.card.description || <span className="muted">(empty)</span>}</p>
        <p className="small">Parameters: {data.card.params.map((p) => `${p.name}${p.description ? '' : ' (no description)'}`).join(', ') || 'none'}</p>
      </div>
      {data.whatif.length > 0 && (
        <div className="card">
          <h4>Measured rewrite effect (results/whatif.jsonl)</h4>
          <table>
            <thead><tr><th>Task</th><th>BM25</th><th>Semantic</th></tr></thead>
            <tbody>{data.whatif.map((w) => (
              <tr key={w.task_id + w.cap_id}><td className="small">{w.task_id.slice(0, 8)}</td><td><Rank pair={w.bm25} /></td><td><Rank pair={w.semantic} /></td></tr>
            ))}</tbody>
          </table>
        </div>
      )}
      {data.tasks.length > 0 ? <WhatIfEditor key={toolId} tool={data} /> : <p className="muted">No task in the answer key needs this tool.</p>}
    </div>
  )
}

export default function ToolHealth() {
  const { data, error, loading } = useLoad(() => api.tools(600), [])
  const [sel, setSel] = useState(() => {
    const h = decodeURIComponent(window.location.hash.slice(1))
    return h.startsWith('tool=') ? h.slice(5).split('&')[0] : null
  })
  const [onlyGold, setOnlyGold] = useState(true)
  if (!data) return <Status loading={loading} error={error} />
  const rows = data.filter((r) => !onlyGold || r.gold_caps > 0)
  return (
    <section className="split">
      <div className="list">
        <h2>Tool health</h2>
        <label className="small"><input type="checkbox" checked={onlyGold} onChange={(e) => setOnlyGold(e.target.checked)} /> only tools some task needs</label>
        <table>
          <thead><tr><th>Tool</th><th title="retrieval-only misses over 6 strategies">Missed</th><th>Flags</th><th>Rewrite</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.tool_id} className={sel === r.tool_id ? 'selected' : ''} onClick={() => setSel(r.tool_id)}>
                <td className="small mono">{r.tool_id}</td>
                <td className="num">{r.missed}</td>
                <td className="small">{r.flags.map((f) => <span key={f} className="tag">{FLAG_NAMES[f] || f}</span>)}</td>
                <td className="small">{r.whatif ? (r.whatif.moved_into_top10.length ? <span className="tag good">top-10</span> : 'no fix') : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="pane">{sel ? <ToolDetail toolId={sel} /> : <p className="muted">Select a tool to see its health and try a new description.</p>}</div>
    </section>
  )
}

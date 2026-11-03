import { useState } from 'react'
import { api } from './api'
import { useLoad } from './hooks'
import { Pct, Status } from './components'

const METRICS = [
  ['recall@5', 'R@5'],
  ['recall@10', 'R@10'],
  ['ndcg@10', 'nDCG@10'],
  ['mrr@10', 'MRR@10'],
  ['completeness@10', 'C@10'],
]

function RetrievalTable() {
  const { data, error, loading } = useLoad(api.metrics, [])
  if (!data) return <Status loading={loading} error={error} />
  const best = Object.fromEntries(
    METRICS.map(([m]) => [m, Math.max(...Object.values(data.strategies).map((s) => s.mean[m]))]),
  )
  return (
    <>
      <table>
        <thead>
          <tr>
            <th>Strategy</th>
            {METRICS.map(([, l]) => <th key={l}>{l}</th>)}
            <th>Δ nDCG@10 vs {data.baseline}</th>
            <th>p (Bonferroni)</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(data.strategies).map(([name, s]) => (
            <tr key={name}>
              <td>{name}{name === data.baseline && <span className="tag">baseline</span>}</td>
              {METRICS.map(([m]) => (
                <td key={m} className={`num ${s.mean[m] === best[m] ? 'best' : ''}`}>{s.mean[m].toFixed(3)}</td>
              ))}
              <td className="num">{name === data.baseline ? '–' : (s.delta['ndcg@10'] >= 0 ? '+' : '') + s.delta['ndcg@10'].toFixed(3)}</td>
              <td className="num">{name === data.baseline ? '–' : s.p_bonf['ndcg@10'].toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">
        {data.n_queries} task questions, 525 tools. Paired tests against {data.baseline}, Bonferroni over {data.n_comparisons} comparisons:
        no strategy differs significantly at α = 0.05.
      </p>
    </>
  )
}

function AgentTable() {
  const runs = useLoad(api.agentRuns, [])
  const success = useLoad(() => api.taskSuccess('status'), [])
  const judge = useLoad(() => api.judge('status'), [])
  if (!runs.data || !success.data || !judge.data)
    return <Status loading={runs.loading || success.loading || judge.loading} error={runs.error || success.error || judge.error} />
  const kimiHuman = judge.data.groups.find((g) => g.labels === 'kimi_bm25_full')
  const sample = judge.data.groups.find((g) => g.labels === 'sample60')
  const ok = (a, s) => success.data.find((r) => r.agent === a && r.strategy === s)
  return (
    <>
      <table>
        <thead>
          <tr>
            <th>Agent</th><th>Strategy</th><th>Task success</th><th>Gold tool used</th>
            <th>All capabilities</th><th>No tool call</th><th>Call error rate</th>
          </tr>
        </thead>
        <tbody>
          {runs.data.map((r) => {
            const s = ok(r.agent, r.strategy)
            const isKimi = r.agent.startsWith('Kimi')
            let cell
            if (isKimi && r.strategy === 'bm25') cell = <><Pct n={kimiHuman.human_success} of={95} /> <span className="tag">human</span></>
            else if (isKimi) cell = <span className="muted" title="The judge failed on Kimi (D-075); no human labels for this run">not validated</span>
            else cell = <><Pct n={s?.success} of={95} /> <span className="tag">judge</span></>
            return (
              <tr key={r.agent + r.strategy}>
                <td>{r.agent}</td><td>{r.strategy}</td><td>{cell}</td>
                <td className="num"><Pct n={r.gold_any} of={r.n} /></td>
                <td className="num"><Pct n={r.gold_all} of={r.n} /></td>
                <td className="num">{r.no_tool}</td>
                <td className="num">{r.calls ? ((100 * r.errs) / r.calls).toFixed(0) + '%' : '–'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="note">
        Task success: LiveMCPEval prompt + call status (D-074), DeepSeek V3.2. Validated on a blind sample of 60 trajectories of the
        other agents (agreement {(100 * sample.agreement).toFixed(1)}%, κ {sample.kappa.toFixed(2)}, 95% CI {sample.kappa_ci[0].toFixed(2)}–{sample.kappa_ci[1].toFixed(2)}).
        It fails on Kimi (κ {kimiHuman.kappa.toFixed(2)}), so Kimi bm25 shows the human labels. “Gold tool used” = a successful call to an
        acceptable tool; a tool-use measure, not task success.
      </p>
    </>
  )
}

function QueryExplorer() {
  const runs = useLoad(api.metrics, [])
  const [taskId, setTaskId] = useState('')
  const [input, setInput] = useState('')
  const q = useLoad(() => (taskId ? api.query(taskId) : Promise.resolve(null)), [taskId])
  const strategies = runs.data ? Object.keys(runs.data.strategies) : []
  const gold = new Set((q.data?.capabilities || []).flatMap((c) => c.acceptable))
  return (
    <>
      <form className="row" onSubmit={(e) => { e.preventDefault(); setTaskId(input.trim()) }}>
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Task id, e.g. 29fa007d-5caf-41a0-b107-e9388305c6c5" />
        <button type="submit">Show top-10</button>
      </form>
      {taskId && <Status loading={q.loading} error={q.error} />}
      {q.data && (
        <>
          <p className="question">{q.data.question}</p>
          <p className="muted small">Gold tools (acceptable): {[...gold].join(', ')}</p>
          <div className="grid">
            {strategies.map((s) => (
              <div key={s} className="card">
                <h4>{s}</h4>
                <ol>
                  {(q.data.top10[s] || []).map((t) => <li key={t} className={gold.has(t) ? 'gold' : ''}>{t}</li>)}
                </ol>
              </div>
            ))}
          </div>
        </>
      )}
    </>
  )
}

export default function Compare() {
  return (
    <section>
      <h2>Which retrieval strategy wins?</h2>
      <RetrievalTable />
      <h2>Agents on the live MCP servers</h2>
      <AgentTable />
      <h2>Per-query view</h2>
      <QueryExplorer />
    </section>
  )
}

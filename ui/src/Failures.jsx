import { useState } from 'react'
import PropTypes from 'prop-types'
import { api } from './api'
import { useLoad } from './hooks'
import { Bar, LABEL_HELP, LABEL_NAMES, Legend, Status } from './components'

const V3_HELP = {
  ...LABEL_HELP,
  'agent selection error': 'An acceptable tool was exposed but not used, or an acceptable call failed. Inspect exposure and call order below.',
  'composition error': 'Not assigned from offline discoverability in v3. Dependency failures were not tested.',
  'ranking confusion': 'On a logged query, another retriever found an acceptable tool within the actual cutoff, or this retriever ranked one below the cutoff but within 20.',
  'bad description': 'Not assigned for agent rows in v3. Description edits were not tested on actual agent searches.',
  'retrieval miss': 'Unresolved on the actual logged queries and cutoffs. This does not establish that the queries were adequate.',
  unassigned: 'The recorded evidence is insufficient to assign a fault. This is a coverage limitation, not a fault category.',
}


function Trajectory({ agent, strategy, taskId }) {
  const { data, error, loading } = useLoad(() => api.trajectory(agent, strategy, taskId), [agent, strategy, taskId])
  if (!data) return <Status loading={loading} error={error} />
  return (
    <div className="trajectory">
      {data.steps.map((s, i) =>
        s.type === 'search' ? (
          <div key={i} className="step search">
            <b>search</b> “{s.query}” → <span className="muted small">{s.shortlist.join(', ')}</span>
          </div>
        ) : (
          <div key={i} className={`step call ${s.ok ? 'ok' : 'err'}`}>
            <b>{s.ok ? 'OK' : 'ERROR'}</b> {s.tool_id}
            <span className="muted small"> {JSON.stringify(s.args).slice(0, 160)}</span>
            {!s.ok && s.error && <div className="small err-text">{s.error.slice(0, 220)}</div>}
            <details className="small"><summary>Arguments and saved result</summary>
              <pre>{JSON.stringify(s.args, null, 2)}</pre>
              <p>Gateway status is recorded separately from result content. Only the excerpt retained in the log is available.</p>
              <pre>{s.result_snippet || '(no saved result text)'}</pre>
              {s.error && <pre>{s.error}</pre>}
            </details>
          </div>
        ),
      )}
      <div className="step final"><b>Agent final answer</b> ({data.stop_reason}) {(data.final_answer || '(empty)').slice(0, 600)}
        {data.final_answer && <details><summary>Full saved final answer</summary><pre>{data.final_answer}</pre></details>}
      </div>
    </div>
  )
}

function RepairEvidence({ assessment }) {
  const status = { rescued: 'Rescue observed', no_rescue: 'Tested; no top-10 rescue', not_tested: 'Not tested' }
  return (
    <details className="small">
      <summary>Description repair: {status[assessment.status]} · {assessment.strategy}</summary>
      <p>Original task question · top-{assessment.cutoff}. These offline probes do not test description changes on actual agent searches.</p>
      {assessment.status === 'not_tested' && <p>No saved probe for this capability under this retriever. No conclusion about repair effectiveness.</p>}
      {assessment.evidence.map((e) => (
        <div key={e.evidence_id} className="card">
          <b>{e.tool_id}</b>: rank {e.before_rank} → {e.after_rank} · {e.rescued ? 'entered the cutoff' : 'no cutoff crossing'}
          <p>Probe query: {e.query}</p>
          <p>Frozen rewrite: {e.intervention.prompt_version} · {e.intervention.model}</p>
          <p>{e.provenance}.</p>
          <p><code>{e.probe_source.path}</code> · line {e.probe_source.line}</p>
          <details><summary>Evidence identifiers</summary>
            <div className="step">Query SHA-256: <code>{e.query_sha256}</code></div>
            <div className="step">Intervention: <code>{e.intervention_id}</code></div>
            <div className="step">Evidence: <code>{e.evidence_id}</code></div>
          </details>
        </div>
      ))}
    </details>
  )
}

RepairEvidence.propTypes = { assessment: PropTypes.object.isRequired }

function SearchEvidence({ evidence, offline, omitted }) {
  return (
    <details className="small">
      <summary>Actual search and call evidence</summary>
      <p>Recorded gateway order is shown below. Assistant-turn grouping was not saved. Exposure does not prove what the model reasoned.</p>
      <p>All query replays available: {evidence.replay_complete ? 'yes' : 'no'} · original shortlists reproduced: {evidence.original_strategy_reproduced ? 'yes' : 'no / unavailable'}.</p>
      {evidence.search_results_truncated && <p>A search response was truncated or its visibility is unknown. Returned IDs alone do not prove that the tool cards reached the agent.</p>}
      {evidence.events.map((e) => (
        <div className="card" key={e.event_id}>
          <b>Search at event {e.step_position} · top-{e.k}</b>
          <p>{e.query}</p>
          <p>Returned tools: {e.logged_shortlist.join(', ')}</p>
          <p>Tool-card visibility: {e.search_result_truncated === false ? 'untruncated response' : 'truncated / unknown'} · acceptable tool exposed: {evidence.exposure_event_ids.includes(e.event_id) ? 'yes' : 'not established'}.</p>
          <p>Replay agreement: {e.original_strategy_reproduced === true ? 'exact shortlist match' : e.original_strategy_reproduced === false ? 'mismatch' : 'unavailable'}.</p>
          <details><summary>Acceptable-tool ranks in replay</summary>
            <p>Only observed ranks within the saved depth are listed. Missing or failed measurements cannot establish a miss.</p>
            <ul>{evidence.replay_hits.filter((h) => h.event_id === e.event_id).map((h) => (
              <li key={h.replay_id}>{h.strategy}: rank {h.rank} · {h.inside_cutoff ? 'inside' : 'outside'} top-{h.k}</li>
            ))}</ul>
          </details>
          <p>Source: <code>{e.source.path}</code> · line {e.source.line}</p>
        </div>
      ))}
      {!evidence.events.length && <p>No recorded search for this task.</p>}
      <details><summary>Call order and prior exposure</summary>
        {evidence.calls.map((c) => (
          <p key={c.step_position}>Event {c.step_position}: {c.tool_id} · {c.outcome}
            {' · '}{c.acceptable ? 'acceptable tool' : 'outside this capability’s answer key'}
            {' · '}acceptable tool exposed before call: {c.acceptable_exposed_before ? 'yes' : 'not established'}</p>
        ))}
      </details>
      <details><summary>Separate v2 offline context</summary>
        <p>{offline.label} · <code>{offline.reason}</code>. Original task question and top-10; this does not establish actual-query repair or a dependency failure.</p>
        {omitted && <p>V2 suggested an omitted step. This hypothesis has not been validated as a composition fault.</p>}
        {offline.repair_assessment && <RepairEvidence assessment={offline.repair_assessment} />}
      </details>
    </details>
  )
}

SearchEvidence.propTypes = {
  evidence: PropTypes.object.isRequired,
  offline: PropTypes.object.isRequired,
  omitted: PropTypes.bool,
}

function CasePage({ agent, strategy, label, reportId, variant, diagnosis, offset, onPage }) {
  const { data, error, loading } = useLoad(
    () => api.cases(agent, strategy, label, { offset, limit: 20, variant, diagnosis, reportId }),
    [agent, strategy, label, reportId, variant, diagnosis, offset],
  )
  const [open, setOpen] = useState(null)
  if (!data) return <Status loading={loading} error={error} />
  return (
    <div>
      <p className="note">
        <b>{LABEL_NAMES[label] || 'Unassigned'}</b>: {(diagnosis === 'v3' ? V3_HELP : LABEL_HELP)[label]}<br />
        {data.total} capabilities in failed tasks — {label === 'unassigned' ? 'excluded from the hypothesis chart because evidence is insufficient' : 'the exact cases counted in this chart segment'}.
        {' '}Task outcomes: {data.failed_by}.
        {data.exploratory && ' Exploratory: these outcomes are outside the D-075 validated reporting scope.'}
      </p>
      <div className="row" aria-label="Case pagination">
        <button disabled={data.offset === 0} onClick={() => onPage(Math.max(0, data.offset - data.limit))}>Previous</button>
        <span role="status">{data.items.length ? `${data.offset + 1}–${data.offset + data.items.length}` : '0'} of {data.total} cases</span>
        <button disabled={data.offset + data.items.length >= data.total} onClick={() => onPage(data.offset + data.limit)}>Next</button>
      </div>
      {data.total === 0 && <p>No cases in this chart segment.</p>}
      {data.items.map((c) => (
        <div key={c.case_id} className="card case" data-case-id={c.case_id}>
          <button className="link" onClick={() => setOpen(open === c.case_id ? null : c.case_id)}>
            {c.question.slice(0, 140)}{c.question.length > 140 ? '…' : ''}
          </button>
          <div className="small">
            Capability <b>{c.capability}</b> · acceptable: {c.acceptable.join(', ')} · rule hypothesis: <code>{c.reason}</code>
          </div>
          <div className="small">
            Task outcome: {c.outcome.verdict} · {c.outcome.source === 'human' ? 'human review' : `${c.outcome.model} (${c.outcome.variant})`}
            {c.outcome.exploratory && ' · exploratory'}
          </div>
          <details className="small">
            <summary>Outcome evidence</summary>
            <p>{c.outcome.validation_scope}</p>
            <p>Recorded verdict: {c.outcome.raw_verdict}. {c.outcome.interpretation}.</p>
            <p><code>{c.outcome.evidence.path}</code> · {c.outcome.evidence.line ? `line ${c.outcome.evidence.line}` : `record ${c.outcome.evidence.record}`}</p>
            <p>Task: <code>{c.task_id}</code> · capability: <code>{c.cap_id}</code></p>
          </details>
          {c.repair_assessment && <RepairEvidence assessment={c.repair_assessment} />}
          {c.search_evidence && <SearchEvidence evidence={c.search_evidence} offline={c.offline_diagnosis} omitted={c.omitted_step_hypothesis} />}
          {open === c.case_id && <Trajectory agent={agent} strategy={strategy} taskId={c.task_id} />}
        </div>
      ))}
    </div>
  )
}

function CaseList(props) {
  const [offset, setOffset] = useState(0)
  // Remount on pagination so a slow response cannot display the previous page.
  return <CasePage key={offset} {...props} offset={offset} onPage={setOffset} />
}

Trajectory.propTypes = {
  agent: PropTypes.string.isRequired,
  strategy: PropTypes.string.isRequired,
  taskId: PropTypes.string.isRequired,
}

CasePage.propTypes = {
  agent: PropTypes.string.isRequired,
  strategy: PropTypes.string.isRequired,
  label: PropTypes.string.isRequired,
  reportId: PropTypes.string.isRequired,
  variant: PropTypes.string.isRequired,
  diagnosis: PropTypes.string.isRequired,
  offset: PropTypes.number.isRequired,
  onPage: PropTypes.func.isRequired,
}

function FailureReport({ diagnosis }) {
  const { data, error, loading } = useLoad(() => api.failures('status', diagnosis), [diagnosis])
  const [pick, setPick] = useState(null)
  if (!data) return <Status loading={loading} error={error} />
  if (data.schema_version !== 2) return <p className="error">This report needs case membership. Run the failure-report exporter with TOOLSCOPE_JUDGE_VARIANT=status, then reload.</p>
  return (
    <section>
      <p className="note">
        One rule-derived hypothesis per unsolved capability with sufficient recorded evidence in failed tasks ({data.classifier_version}, deterministic, from logs).
        Failed = human labels where they exist, otherwise the judge. Kimi semantic outcomes are exploratory (D-075).
        Click a segment to inspect its counted cases. These hypotheses have not undergone independent expert validation.
      </p>
      <p className="note">
        {diagnosis === 'v3'
          ? 'Agent diagnoses use actual logged queries, actual cutoffs and chronological exposure. Unassigned cases are shown separately. Query adequacy, description repair on actual searches and dependency failures are not established.'
          : diagnosis === 'v2'
          ? 'Description repair requires a measured rescue under the same retriever on the original task question at top-10. Agent rows use this offline proxy; actual agent searches/top-5 remain untested.'
          : 'Historical v1: a rescue under either tested retriever was reused across strategies. These labels are preserved for comparison and do not establish strategy-specific repairability.'}
      </p>
      {diagnosis === 'v3' && <p className="note">
        Replay: {data.repair_scope.replay_summary.measurements} measurements over {data.repair_scope.replay_summary.unique_queries} distinct queries.
        {' '}{data.repair_scope.replay_summary.original_strategy_matches} of {data.repair_scope.replay_summary.search_events} original shortlists reproduced;
        {' '}{data.repair_scope.replay_summary.errors} replay errors;
        {' '}{data.repair_scope.replay_summary.truncated_search_results} recorded responses truncated.
      </p>}
      <Legend help={diagnosis === 'v3' ? V3_HELP : LABEL_HELP} />
      <table className="bars">
        <tbody>
          {data.agent.map((r) => (
            <tr key={r.agent + r.strategy}>
              <td className="runname">{r.agent}<span className="muted"> · {r.strategy}</span>
                <div className="muted small">{r.failed_tasks} failed tasks · {r.total_labels} labels</div>
                <div className="muted small">failed by: {r.failed_by}{r.exploratory && ' · exploratory'}</div>
                <div className="muted small">{r.failed_tasks_without_labels} failed tasks have no assigned hypothesis</div>
                {diagnosis === 'v3' && <>
                  <button className="link small" disabled={!r.unassigned_capabilities}
                          onClick={() => setPick({ agent: r.agent, strategy: r.strategy, label: 'unassigned' })}>
                    {r.unassigned_capabilities} unassigned capabilities · inspect coverage
                  </button>
                  <div className="muted small">{r.failed_tasks_only_unassigned} tasks have only unassigned capabilities; {r.failed_tasks_all_capabilities_solved} have all acceptable calls completed</div>
                </>}
                {r.missing_outcomes > 0 && <div className="muted small">{r.missing_outcomes} tasks have no outcome and are excluded</div>}
              </td>
              <td className="barcell">
                <Bar counts={r.labels} total={r.total_labels}
                     picked={pick && pick.agent === r.agent && pick.strategy === r.strategy ? pick.label : null}
                     onPick={(label) => setPick({ agent: r.agent, strategy: r.strategy, label })} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {pick && (
        <>
          <h3>{pick.agent} · {pick.strategy}</h3>
          <CaseList key={[data.report_id, pick.agent, pick.strategy, pick.label].join('|')}
                    {...pick} reportId={data.report_id} variant={data.judge_variant} diagnosis={diagnosis} />
        </>
      )}
      <h3>Retrieval only (task question, top-10, 255 steps per strategy)</h3>
      {diagnosis !== 'v1' && <p className="note">Repair probes exist for {data.repair_scope.tested_repair_strategies.join(' and ')} only. “Not tested” is not a negative result.</p>}
      <table>
        <thead><tr><th>Strategy</th><th>Found</th><th>Ranking confusion</th><th>Description-repairable</th><th>Retrieval miss</th></tr></thead>
        <tbody>
          {data.retrieval.map((r) => (
            <tr key={r.strategy}>
              <td>{r.strategy}</td>
              {['found', 'ranking confusion', 'bad description', 'retrieval miss'].map((l) => <td key={l} className="num">
                {diagnosis !== 'v1' && l === 'bad description' && !data.repair_scope.tested_repair_strategies.includes(r.strategy)
                  ? 'Not tested' : r.labels[l] || 0}
              </td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

FailureReport.propTypes = { diagnosis: PropTypes.string.isRequired }

export default function Failures() {
  const [diagnosis, setDiagnosis] = useState('v3')
  return (
    <section>
      <h2>Failure evidence and hypotheses</h2>
      <p className="note">Inspect the evidence behind each rule-based hypothesis. The categories support investigation; they do not establish a validated causal assignment of fault. Independent diagnosis review was not completed.</p>
      <label>Evidence version: <select value={diagnosis} onChange={(e) => setDiagnosis(e.target.value)}>
        <option value="v3">v3 · actual agent searches and call order</option>
        <option value="v2">v2 · strategy-specific repair evidence</option>
        <option value="v1">v1 · historical pooled evidence</option>
      </select></label>
      <FailureReport key={diagnosis} diagnosis={diagnosis} />
    </section>
  )
}

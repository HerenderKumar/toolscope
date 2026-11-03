import PropTypes from 'prop-types'

export const LABELS = [
  'environment failure',
  'agent selection error',
  'composition error',
  'ranking confusion',
  'bad description',
  'retrieval miss',
]

// Paper wording for "bad description" (D-073 / positioning note): a counterfactual, not a unique cause.
export const LABEL_NAMES = {
  'environment failure': 'Environment',
  'agent selection error': 'Agent selection',
  'composition error': 'Composition',
  'ranking confusion': 'Ranking confusion',
  'bad description': 'Description-repairable',
  'retrieval miss': 'Retrieval miss',
}

export const LABEL_HELP = {
  'environment failure': 'Every acceptable tool is broken upstream, or the gold call failed for an outside reason.',
  'agent selection error': 'An acceptable tool was in the agent’s shortlist (or the task question finds it) but was not used successfully.',
  'composition error': 'Another step of the task was solved, but this step was dropped although retrieval could serve it.',
  'ranking confusion': 'This strategy misses the tool, but another strategy ranks it in its top-10, or it sits at rank 11–20.',
  'bad description': 'An observed description-repair result. Inspect the tested strategy, query and cutoff; legacy v1 pooled evidence across strategies.',
  'retrieval miss': 'An unresolved retrieval miss. An untested rewrite is not evidence that rewriting cannot help.',
}

export function Status({ loading, error }) {
  if (loading) return <p className="muted">Loading…</p>
  if (error) return <p className="error">Could not load: {error.message}</p>
  return null
}

export function Pct({ n, of }) {
  if (n == null || !of) return <span className="muted">–</span>
  return (
    <span>
      {n} <span className="muted">({((100 * n) / of).toFixed(1)}%)</span>
    </span>
  )
}

export function Bar({ counts, total, onPick, picked }) {
  return (
    <div className="bar" role="img" aria-label="rule-derived hypothesis shares">
      {LABELS.map((l, i) => {
        const n = counts[l] || 0
        if (!n) return null
        return (
          <button
            key={l}
            className={`seg seg-${i} ${picked === l ? 'picked' : ''}`}
            style={{ flexGrow: n }}
            title={`${LABEL_NAMES[l]}: ${n} of ${total}`}
            onClick={() => onPick && onPick(l)}
          >
            {n / total >= 0.07 ? n : ''}
          </button>
        )
      })}
    </div>
  )
}

export function Legend({ help = LABEL_HELP }) {
  return (
    <div className="legend">
      {LABELS.map((l, i) => (
        <span key={l} title={help[l]}>
          <i className={`swatch seg-${i}`} /> {LABEL_NAMES[l]}
        </span>
      ))}
    </div>
  )
}

Legend.propTypes = { help: PropTypes.object }

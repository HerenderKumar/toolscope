"""D-079: export diagnosis v3 after a complete, fingerprint-matched local replay.

Run: PYTHONPATH=src .venv/bin/python scripts/diagnose_agent_searches.py
Reads saved evidence only. Keeps original v1/v2 artifacts unchanged.
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from toolscope.diagnosis.search_evidence import annotate_events, collect, diagnose, query_id
from toolscope.retrievers.base import STRATEGIES

OUT = Path('results/diagnosis_v3')


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_replay(manifest: dict, summary: dict, rows: list[dict], events: list[dict]) -> dict:
    """Reject interrupted or mixed measurements before producing any diagnostic output."""
    replay = {}
    for row in rows:
        if row['fingerprint'] != manifest['fingerprint'] or row['query_id'] != query_id(row['query']):
            raise ValueError('Mixed replay fingerprints or query identities')
        if row['status'] == 'ok' and (row['fallback_used'] or row['depth'] < 20
                or len(row['top_k']) < row['depth']
                or len({t for t, _ in row['top_k']}) != len(row['top_k'])):
            raise ValueError('Invalid successful replay ranking')
        replay[(row['strategy'], row['query_id'])] = row
    queries = {e['query_id'] for e in events if e['replay_eligible']}
    expected = {(s, q) for q in queries for s in STRATEGIES}
    if replay.keys() != expected:
        raise ValueError('Incomplete replay measurement set')
    checks = {'fingerprint': manifest['fingerprint'], 'measurements': len(replay),
              'unique_queries': len(queries), 'search_events': len(events),
              'errors': sum(r['status'] != 'ok' for r in replay.values()),
              'original_strategy_matches': sum(e['original_strategy_reproduced'] is True for e in events),
              'original_strategy_mismatches': sum(e['original_strategy_reproduced'] is False for e in events),
              'original_strategy_unavailable': sum(e['original_strategy_reproduced'] is None for e in events)}
    if any(summary.get(k) != v for k, v in checks.items()):
        raise ValueError('Replay summary does not match saved measurements')
    if annotate_events(events, replay) != events:
        raise ValueError('Saved event agreement does not match replay')
    return replay


def main() -> None:
    manifest = json.loads((OUT / 'replay_manifest.json').read_text())
    for path, sha in manifest['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Replay source changed: {path}')
    summary = json.loads((OUT / 'replay_summary.json').read_text())
    events = read(OUT / 'search_events.jsonl')
    replay = validate_replay(manifest, summary, read(OUT / 'replay.jsonl'), events)
    runs, current_events = collect()
    if annotate_events(current_events, replay) != events:
        raise ValueError('Final trajectories changed since replay')
    grouped = defaultdict(list)
    for event in events:
        grouped[(event['agent'], event['strategy'], event['task_id'])].append(event)
    gold_path = Path('data/gold/gold.jsonl')
    gold = {r['task_id']: r for r in read(gold_path)}
    prior_path = Path('results/diagnosis_v2/failure_labels.jsonl')
    prior_manifest_path = Path('results/diagnosis_v2/manifest.json')
    prior_manifest = json.loads(prior_manifest_path.read_text())
    prior = read(prior_path)
    rows = []
    for row in prior:
        if row['scope'] == 'retrieval':
            rows.append(row)
            continue
        task = gold[row['task_id']]
        cap = next(c for c in task['capabilities'] if c['cap_id'] == row['cap_id'])
        record = runs[(row['agent'], row['strategy'])][row['task_id']]
        searches = grouped[(row['agent'], row['strategy'], row['task_id'])]
        result = diagnose(row, cap, record, searches, replay)
        # Save the exact supporting events with the case, so the UI needs no joins or recomputation.
        result['search_evidence']['events'] = searches
        rows.append(result)
    changes = [{'scope': r['scope'], 'agent': r['agent'], 'strategy': r['strategy'], 'task_id': r['task_id'],
                'cap_id': r['cap_id'], 'before_label': p['label'], 'after_label': r['label'],
                'before_reason': p['reason'], 'after_reason': r['reason']}
               for p, r in zip(prior, rows) if p['label'] != r['label']]
    counts = Counter(r.get('diagnostic_status', 'offline_retrieval') for r in rows)
    manifest.update(classifier_version='D-079', agent_query_kind='observed_agent_search',
                    offline_classifier_version='D-078', query_replay_complete=summary['errors'] == 0,
                    actual_query_repair_tested=False,
                    tested_repair_strategies=prior_manifest['tested_repair_strategies'],
                    repair_query_kind='offline_task_question',
                    chronology_limit='Recorded gateway event order; assistant-turn grouping was not saved',
                    label_rows=len(rows), changed_labels=len(changes), coverage=dict(counts), replay_summary=summary)
    inputs = [gold_path, prior_path, prior_manifest_path, OUT/'replay.jsonl', OUT/'search_events.jsonl',
              OUT/'replay_summary.json', Path(__file__), Path('src/toolscope/diagnosis/search_evidence.py'),
              Path('src/toolscope/diagnosis/classify.py'), Path('src/toolscope/agent/watch.py')]
    manifest['source_sha256'].update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    lines = ['# Diagnosis v3: actual agent-search evidence (D-079)', '',
             'Original logged shortlists establish returned tools; only untruncated responses establish observed exposure.',
             'Recorded event order is available, but assistant-turn grouping is not. Earlier exposure is not proof of reasoning.',
             'Replay compares retrievers on the exact logged query and k. It does not judge query adequacy or replay descriptions.',
             '', '| Agent | Strategy | Assigned unsolved capabilities | Unassigned | Acceptable call succeeded |',
             '|---|---|---:|---:|---:|']
    for agent, strategy in runs:
        selected = [r for r in rows if r['scope'] == 'agent' and r['agent'] == agent and r['strategy'] == strategy]
        lines.append(f"| {agent} | {strategy} | {sum(r['diagnostic_status'] == 'assigned' for r in selected)} | "
                     f"{sum(r['label'] is None for r in selected)} | {sum(r['label'] == 'solved' for r in selected)} |")
    lines += ['', 'Counts above include all tasks. Failure reports separately apply unchanged human/judge task outcomes.',
              f'Changed labels: {len(changes)}; details in changes.json. Null labels are coverage limitations, not a seventh fault category.',
              'V2 task-question diagnoses remain nested as offline context. No actual-query composition or description-repair claim is inferred from them.']
    (OUT / 'failure_labels.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (OUT / 'changes.json').write_text(json.dumps(changes, indent=1) + '\n')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    (OUT / 'changes.md').write_text('\n'.join(lines) + '\n')
    print(f"rows={len(rows)} events={len(events)} changed_labels={len(changes)} "
          f"unassigned={sum(r['label'] is None for r in rows)} out={OUT}")


if __name__ == '__main__':
    main()

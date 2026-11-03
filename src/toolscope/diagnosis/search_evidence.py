"""D-079: observed search events and chronological exposure, separate from offline proxies."""

import hashlib
import json
from pathlib import Path
from typing import Any

from toolscope.diagnosis.classify import call_outcome, ENVIRONMENT, SELECTION, RANKING, RETRIEVAL
from toolscope.diagnosis.runs import RUNS, load
from toolscope.retrievers.base import STRATEGIES

VERSION = "D-079"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def query_id(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()


def collect(root: Path = Path('.')) -> tuple[dict, list[dict]]:
    """Retain final-run override order and exact source references; event positions are 1-based."""
    runs, events = {}, []
    for (agent, strategy), folders in RUNS.items():
        records = load(agent, strategy, folders, root=root)
        sources = {}
        for folder in folders:
            path = Path(folder) / 'agent.jsonl'
            if not (root / path).exists():
                continue
            for line, raw in enumerate((root / path).read_text().splitlines(), 1):
                rec = json.loads(raw)
                if rec.get('strategy') == strategy and rec == records.get(rec['task_id']):
                    sources[rec['task_id']] = {'path': str(path), 'line': line, 'record_sha256': digest(rec)}
        runs[(agent, strategy)] = records
        for task_id, rec in sorted(records.items()):
            for pos, step in enumerate(rec['steps'], 1):
                if step.get('type') != 'search':
                    continue
                q = step.get('query')
                k = rec.get('k')
                valid = isinstance(q, str) and bool(q) and type(k) is int and k > 0
                events.append({'event_id': digest([agent, strategy, task_id, sources[task_id]['record_sha256'], pos]),
                               'agent': agent, 'strategy': strategy, 'task_id': task_id, 'step_position': pos,
                               'query': q, 'query_id': query_id(q) if isinstance(q, str) else None,
                               'k': k, 'k_source': 'trajectory.k', 'logged_shortlist': step.get('shortlist', []),
                               'search_result_truncated': step.get('truncated'),
                               'replay_eligible': valid, 'source': sources[task_id]})
    return runs, events


def annotate_events(events: list[dict], replay: dict[tuple[str, str], dict]) -> list[dict]:
    out = []
    for event in events:
        own = replay.get((event['strategy'], event['query_id']))
        comparable = (event['replay_eligible'] and own is not None and own['status'] == 'ok'
                      and own['depth'] >= event['k'])
        matches = ([t for t, _ in own['top_k'][:event['k']]] == event['logged_shortlist']) if comparable else None
        out.append({**event, 'original_strategy_reproduced': matches,
                    'replay_ids': {s: replay[(s, event['query_id'])]['replay_id'] for s in STRATEGIES
                                   if (s, event['query_id']) in replay}})
    return out


def capability_evidence(cap: dict, record: dict, events: list[dict], replay: dict) -> dict:
    acceptable = set(cap['acceptable'])
    returned = [e for e in events if acceptable.intersection(e['logged_shortlist'])]
    exposures = [e for e in returned if e['search_result_truncated'] is False]
    calls = []
    for position, step in enumerate(record['steps'], 1):
        if step.get('type') != 'call':
            continue
        earlier = [e for e in exposures if e['step_position'] < position]
        calls.append({'step_position': position, 'tool_id': step['tool_id'], 'acceptable': step['tool_id'] in acceptable,
                      'outcome': call_outcome(step), 'acceptable_exposed_before': bool(earlier),
                      'exposure_event_ids': [e['event_id'] for e in earlier],
                      'last_shortlist': step.get('shortlist_before', []),
                      'chosen_tool_in_last_shortlist': step.get('in_shortlist')})
    replay_hits = []
    complete = bool(events)
    concordant = bool(events)
    for event in events:
        concordant = concordant and event['original_strategy_reproduced'] is True
        for strategy in STRATEGIES:
            measured = replay.get((strategy, event['query_id']))
            usable = (event['replay_eligible'] and measured is not None and measured['status'] == 'ok'
                      and measured['depth'] >= event['k'])
            complete = complete and usable
            if not usable:
                continue
            rank = next((i for i, (t, _) in enumerate(measured['top_k'], 1) if t in acceptable), None)
            if rank is not None:
                replay_hits.append({'event_id': event['event_id'], 'replay_id': measured['replay_id'],
                                    'step_position': event['step_position'], 'strategy': strategy, 'rank': rank,
                                    'k': event['k'], 'inside_cutoff': rank <= event['k']})
    return {'query_kind': 'observed_agent_search', 'search_event_ids': [e['event_id'] for e in events],
            'returned_event_ids': [e['event_id'] for e in returned],
            'exposure_event_ids': [e['event_id'] for e in exposures], 'exposure_status': 'exposed' if exposures else 'never_exposed',
            'calls': calls, 'replay_hits': replay_hits, 'replay_complete': bool(complete),
            'original_strategy_reproduced': bool(concordant),
            'search_results_truncated': any(e['search_result_truncated'] is not False for e in events),
            'interpretation': 'Logged shortlist membership is observed; replay is a new local measurement, not agent behavior.'}


def diagnose(previous: dict, cap: dict, record: dict, events: list[dict], replay: dict) -> dict:
    """No offline task-question rank can establish an actual-query fault in v3."""
    evidence = capability_evidence(cap, record, events, replay)
    calls = [c for c in evidence['calls'] if c['acceptable']]
    exposed = bool(evidence['exposure_event_ids'])
    result = {**previous, 'classifier_version': VERSION,
              'offline_diagnosis': {k: previous[k] for k in ('label', 'reason', 'repair_assessment')},
              'search_evidence': evidence, 'diagnostic_status': 'assigned',
              'omitted_step_hypothesis': previous['label'] == 'composition error'}
    # The v2 repair assessment remains only inside the separately labelled offline context.
    result.pop('repair_assessment', None)
    if previous['label'] == 'solved':
        result.update(label='solved', reason=previous['reason'], diagnostic_status='acceptable_call_succeeded')
    elif previous['label'] == ENVIRONMENT:
        result.update(label=ENVIRONMENT, reason=previous['reason'])
    elif calls:
        before_call = any(c['acceptable_exposed_before'] for c in calls)
        reason = ('bad_call_after_exposure' if before_call else 'exposed_after_failed_call_not_retried' if exposed
                  else 'bad_call_without_prior_exposure')
        result.update(label=SELECTION, reason=reason)
    elif exposed and not evidence['search_results_truncated']:
        result.update(label=SELECTION, reason='exposed_not_called')
    elif not events:
        result.update(label=None, reason='no_observed_search', diagnostic_status='insufficient_search_evidence')
    elif evidence['search_results_truncated']:
        result.update(label=None, reason='search_result_truncated', diagnostic_status='insufficient_exposure_evidence')
    elif not evidence['replay_complete']:
        result.update(label=None, reason='replay_missing_or_failed', diagnostic_status='insufficient_replay_evidence')
    elif not evidence['original_strategy_reproduced']:
        result.update(label=None, reason='original_shortlist_not_reproduced', diagnostic_status='replay_mismatch')
    else:
        alternatives = [h for h in evidence['replay_hits'] if h['strategy'] != previous['strategy'] and h['inside_cutoff']]
        nearby = [h for h in evidence['replay_hits'] if h['strategy'] == previous['strategy'] and h['k'] < h['rank'] <= 20]
        result.update(label=RANKING if alternatives or nearby else RETRIEVAL,
                      reason='alternative_retriever_found_logged_query' if alternatives else 'logged_query_rank_within_20'
                      if nearby else 'all_replayed_strategies_miss_logged_cutoffs')
    return result

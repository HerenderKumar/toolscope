"""Pure offline repair-study measurements; descriptions must be frozen before tasks are loaded."""

from copy import deepcopy
from typing import Any, Callable

import bm25s
import numpy as np

from toolscope.catalog.build_catalog import make_search_text
from toolscope.retrievers.bm25 import _tokenize
from toolscope.retrievers.runlog import schema_tokens


def edited_cards(cards: list[dict], edits: dict[str, str]) -> list[dict]:
    if not edits.keys() <= {c['tool_id'] for c in cards}:
        raise ValueError('Unknown edited tool')
    out = deepcopy(cards)
    for card in out:
        if card['tool_id'] in edits:
            card['description'] = edits[card['tool_id']]
            card['search_text'] = make_search_text(card['server_name'], card['tool_name'], card['description'], card['params'])
    return out


def bm25_scores(cards: list[dict], queries: list[str]) -> np.ndarray:
    model = bm25s.BM25()
    model.index(_tokenize([c['search_text'] for c in cards]), show_progress=False)
    rows = []
    for tokens in _tokenize(queries):
        known = [t for t in tokens if t in model.vocab_dict]
        rows.append(model.get_scores(known) if known else np.zeros(len(cards)))
    return np.stack(rows)


def ordered(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, axis=1, kind='stable')


def cutoff_evidence(ids: list[str], capabilities: list[dict], k: int, before_ids: list[str],
                    edited_ids: set[str]) -> dict[str, Any]:
    now, before = set(ids[:k]), set(before_ids[:k])
    found = {c['cap_id'] for c in capabilities if now.intersection(c['acceptable'])}
    was = {c['cap_id'] for c in capabilities if before.intersection(c['acceptable'])}
    collateral = {c['cap_id'] for c in capabilities if not edited_ids.intersection(c['acceptable'])}
    return {'found_cap_ids': sorted(found), 'found_capabilities': len(found), 'total_capabilities': len(capabilities),
            'complete': len(found) == len(capabilities), 'complete_before': len(was) == len(capabilities),
            'gained_cap_ids': sorted(found - was), 'lost_cap_ids': sorted(was - found),
            'collateral_gained_cap_ids': sorted((found - was) & collateral),
            'collateral_lost_cap_ids': sorted((was - found) & collateral)}


def measure(task: dict, order: np.ndarray, scores: np.ndarray, cards: list[dict], strategy: str,
            scenario: dict, before_ids: list[str], target_ids: list[str],
            count_tokens: Callable[[str], int]) -> dict:
    ids = [cards[i]['tool_id'] for i in order]
    ranks = {t: i for i, t in enumerate(ids, 1)}
    by_id = {c['tool_id']: c for c in cards}
    cutoffs = {}
    for k in (5, 10):
        evidence = cutoff_evidence(ids, task['capabilities'], k, before_ids, set(scenario['edited_ids']))
        evidence.update(context_tokens=schema_tokens(by_id, ids[:k], count_tokens),
                        description_characters=sum(len(by_id[t]['description']) for t in ids[:k]))
        cutoffs[str(k)] = evidence
    from toolscope.diagnosis.repair_controls import sha
    return {'query_id': sha(task['question']), 'task_id': task['task_id'], 'strategy': strategy,
            'k': 10, 'top_k': [[cards[i]['tool_id'], float(scores[i])] for i in order[:10]],
            'gold_rank': None, 'verdict': None, 'latency_ms': None, 'tokens': cutoffs['10']['context_tokens'],
            'fallback_used': False, **scenario, 'target_ranks': {t: ranks[t] for t in target_ids}, 'cutoffs': cutoffs}


def aggregate(rows: list[dict]) -> list[dict]:
    out = []
    groups = sorted({(r['scenario_id'], r['strategy']) for r in rows})
    for scenario_id, strategy in groups:
        group = [r for r in rows if r['scenario_id'] == scenario_id and r['strategy'] == strategy]
        for k in ('5', '10'):
            values = [r['cutoffs'][k] for r in group]
            n = len(values)
            out.append({'scenario_id': scenario_id, 'strategy': strategy, 'condition': group[0]['condition'],
                        'mode': group[0]['mode'], 'tool_id': group[0]['tool_id'], 'k': int(k), 'tasks': n,
                        'capabilities_found': sum(v['found_capabilities'] for v in values),
                        'capabilities_total': sum(v['total_capabilities'] for v in values),
                        'complete_tasks': sum(v['complete'] for v in values),
                        'newly_complete_tasks': sum(v['complete'] and not v['complete_before'] for v in values),
                        'lost_complete_tasks': sum(v['complete_before'] and not v['complete'] for v in values),
                        **{key: sum(len(v[key]) for v in values) for key in
                           ['gained_cap_ids', 'lost_cap_ids', 'collateral_gained_cap_ids', 'collateral_lost_cap_ids']},
                        'context_tokens_mean': sum(v['context_tokens'] for v in values) / n,
                        'description_characters_mean': sum(v['description_characters'] for v in values) / n})
    return out

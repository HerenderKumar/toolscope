from copy import deepcopy
import sys
from pathlib import Path

from toolscope.diagnosis.search_evidence import annotate_events, capability_evidence, diagnose
from toolscope.retrievers.base import STRATEGIES

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from replay_agent_searches import replay_one  # noqa: E402

CAP = {'cap_id': 't#1', 'acceptable': ['s/gold']}
OLD = {'scope': 'agent', 'agent': 'A', 'task_id': 't', 'cap_id': 't#1', 'strategy': 'bm25',
       'label': 'retrieval miss', 'reason': 'whatif_not_tested', 'repair_assessment': {'status': 'not_tested'}}


def event(position=1, shortlist=None, truncated=False):
    return {'event_id': f'e{position}', 'query_id': 'q', 'step_position': position, 'strategy': 'bm25',
            'k': 1, 'logged_shortlist': shortlist or ['s/other'], 'replay_eligible': True,
            'search_result_truncated': truncated, 'original_strategy_reproduced': True}


def call(tool='s/gold', ok=False):
    return {'type': 'call', 'tool_id': tool, 'ok': ok, 'error': 'missing required argument', 'shortlist_before': []}


def replays():
    return {(s, 'q'): {'status': 'ok', 'replay_id': s, 'depth': 20, 'top_k': [['s/other', 1.0]]}
            for s in STRATEGIES}


def test_later_search_never_explains_an_earlier_bad_call():
    record = {'steps': [call(), {'type': 'search'}]}
    events = [event(2, ['s/gold'])]
    result = diagnose(OLD, CAP, record, events, replays())
    assert result['reason'] == 'exposed_after_failed_call_not_retried'
    assert result['search_evidence']['calls'][0]['acceptable_exposed_before'] is False
    assert result['search_evidence']['calls'][0]['exposure_event_ids'] == []


def test_earlier_exposure_and_history_survive_an_intervening_search():
    record = {'steps': [{'type': 'search'}, {'type': 'search'}, call()]}
    events = [event(1, ['s/gold']), event(2)]
    result = diagnose(OLD, CAP, record, events, replays())
    assert result['reason'] == 'bad_call_after_exposure'
    assert result['search_evidence']['calls'][0]['exposure_event_ids'] == ['e1']


def test_returned_in_truncated_response_does_not_establish_exposure():
    events = [event(1, ['s/gold'], truncated=True)]
    record = {'steps': [{'type': 'search'}]}
    result = diagnose(OLD, CAP, record, events, replays())
    assert result['label'] is None and result['diagnostic_status'] == 'insufficient_exposure_evidence'
    assert result['search_evidence']['returned_event_ids'] == ['e1']
    assert result['search_evidence']['exposure_event_ids'] == []


def test_no_search_never_inherits_offline_composition_or_description_label():
    for label in ('composition error', 'bad description', 'agent selection error'):
        result = diagnose({**OLD, 'label': label}, CAP, {'steps': []}, [], {})
        assert result['label'] is None and result['reason'] == 'no_observed_search'
        assert result['offline_diagnosis']['label'] == label
        assert 'repair_assessment' not in result


def test_actual_k_not_offline_top10_determines_alternative_discovery():
    replay = replays()
    replay[('semantic', 'q')]['top_k'].append(['s/gold', 0.5])  # rank 2, k=1: still missed
    result = diagnose(OLD, CAP, {'steps': [{'type': 'search'}]}, [event()], replay)
    assert result['label'] == 'retrieval miss'
    replay[('semantic', 'q')]['top_k'].reverse()
    result = diagnose(OLD, CAP, {'steps': [{'type': 'search'}]}, [event()], replay)
    assert result['label'] == 'ranking confusion' and result['reason'] == 'alternative_retriever_found_logged_query'


def test_errors_and_unreproduced_shortlists_are_unassigned_not_misses():
    replay = replays(); replay[('semantic', 'q')]['status'] = 'error'
    result = diagnose(OLD, CAP, {'steps': []}, [event()], replay)
    assert result['label'] is None and result['diagnostic_status'] == 'insufficient_replay_evidence'
    result = diagnose(OLD, CAP, {'steps': []}, [{**event(), 'original_strategy_reproduced': False}], replays())
    assert result['label'] is None and result['diagnostic_status'] == 'replay_mismatch'


def test_observed_selection_does_not_depend_on_counterfactual_replay():
    result = diagnose(OLD, CAP, {'steps': []}, [event(1, ['s/gold'])], {})
    assert result['label'] == 'agent selection error' and result['reason'] == 'exposed_not_called'
    for label in ('solved', 'environment failure'):
        result = diagnose({**OLD, 'label': label}, CAP, {'steps': []}, [], {})
        assert result['label'] == label


def test_agreement_compares_exact_order_and_missing_k_is_not_inferred():
    replay = replays()
    rows = annotate_events([event(), event(2, ['s/gold']), {**event(3), 'replay_eligible': False}], replay)
    assert [r['original_strategy_reproduced'] for r in rows] == [True, False, None]


def test_replay_failure_records_bm25_downgrade_without_counting_it_as_success():
    class Broken:
        def retrieve(self, q, k): raise RuntimeError('model unavailable')
    class BM25:
        def retrieve(self, q, k): return [('s/other', 1.0)]
    rec = replay_one(Broken(), 'semantic', 'query', 1, 1, 'f', BM25())
    assert rec['status'] == 'error' and rec['fallback_used'] is True
    assert rec['top_k'] == [] and rec['fallback_top_k'] == [('s/other', 1.0)]
    assert rec['tokens'] is None


def test_bad_acceptable_call_is_observed_without_any_search():
    result = diagnose(OLD, CAP, {'steps': [call()]}, [], {})
    assert result['label'] == 'agent selection error'
    assert result['reason'] == 'bad_call_without_prior_exposure'


def test_repeated_query_keeps_each_exposure_position_and_own_rank_evidence():
    events = [event(1), event(3)]
    replay = replays()
    replay[('bm25', 'q')]['top_k'].append(['s/gold', 0.5])
    result = diagnose(OLD, CAP, {'steps': []}, events, replay)
    assert result['reason'] == 'logged_query_rank_within_20'
    assert [h['step_position'] for h in result['search_evidence']['replay_hits']] == [1, 3]
    assert result['search_evidence']['search_event_ids'] == ['e1', 'e3']


def test_export_rejects_incomplete_mixed_or_inconsistent_replay():
    import pytest
    from diagnose_agent_searches import validate_replay
    from toolscope.diagnosis.search_evidence import query_id

    qid = query_id('query')
    rows = [{'strategy': s, 'query': 'query', 'query_id': qid, 'fingerprint': 'f',
             'status': 'ok', 'fallback_used': False, 'depth': 20, 'replay_id': s,
             'top_k': [[f's/tool{i}', 1.0] for i in range(20)]} for s in STRATEGIES]
    replay = {(r['strategy'], qid): r for r in rows}
    events = annotate_events([{**event(shortlist=['s/tool0']), 'query_id': qid}], replay)
    summary = {'fingerprint': 'f', 'measurements': 6, 'unique_queries': 1, 'search_events': 1,
               'errors': 0, 'original_strategy_matches': 1, 'original_strategy_mismatches': 0,
               'original_strategy_unavailable': 0}
    manifest = {'fingerprint': 'f'}
    assert validate_replay(manifest, summary, rows, events) == replay
    with pytest.raises(ValueError, match='Incomplete'):
        validate_replay(manifest, summary, rows[:-1], events)
    with pytest.raises(ValueError, match='fingerprints'):
        validate_replay({'fingerprint': 'changed'}, summary, rows, events)
    with pytest.raises(ValueError, match='summary'):
        validate_replay(manifest, {**summary, 'errors': 1}, rows, events)
    fallback = deepcopy(rows); fallback[1]['fallback_used'] = True
    with pytest.raises(ValueError, match='Invalid successful'):
        validate_replay(manifest, summary, fallback, events)
    failed = deepcopy(rows); failed[1].update(status='error', top_k=[])
    assert validate_replay(manifest, {**summary, 'errors': 1}, failed, events)[('semantic', qid)]['status'] == 'error'

from copy import deepcopy

import numpy as np
import pytest

from toolscope.diagnosis.repair_controls import condition_rows, needs_translation, translation_messages
from toolscope.diagnosis.repair_study import aggregate, cutoff_evidence, edited_cards, ordered


def card(tid='s/a', desc='old'):
    return {'tool_id': tid, 'server_name': 'S', 'tool_name': 'a', 'description': desc, 'params': [],
            'search_text': 'S | a | old', 'input_schema': {'type': 'object', 'properties': {}}}


def test_translation_input_is_exactly_original_description_and_no_task_context():
    description = '获取卡牌图片'
    assert translation_messages(description)[1] == {'role': 'user', 'content': description}
    assert needs_translation(description) and not needs_translation('Get the card image')


def test_condition_freeze_requires_complete_unique_translations_and_matching_original():
    cards = [card(desc='获取图片')]
    rewrites = [{'tool_id': 's/a', 'original': '获取图片', 'rewritten': 'Retrieve an image',
                 'model': 'saved-model', 'prompt_version': 'saved-prompt'}]
    result = condition_rows(cards, rewrites, {'s/a': 'Get an image'})
    assert result[0]['translation_applied'] and result[0]['structured'] == 'Retrieve an image'
    with pytest.raises(ValueError, match='Incomplete'):
        condition_rows(cards, rewrites, {})
    with pytest.raises(ValueError, match='duplicate'):
        condition_rows(cards, rewrites + rewrites, {'s/a': 'Get an image'})
    with pytest.raises(ValueError, match='untranslated'):
        condition_rows(cards, rewrites, {'s/a': '获取图片'})
    with pytest.raises(ValueError, match='changed'):
        condition_rows([card()], rewrites, {'s/a': 'Get an image'})


def test_english_translation_condition_is_identity_and_schema_never_changes():
    cards = [card()]; before = deepcopy(cards)
    rows = condition_rows(cards, [{'tool_id': 's/a', 'original': 'old', 'rewritten': 'new',
                                  'model': 'm', 'prompt_version': 'p'}], {})
    assert rows[0]['translation'] == 'old' and not rows[0]['translation_applied']
    changed = edited_cards(cards, {'s/a': 'new'})
    assert cards == before and changed[0]['input_schema'] == cards[0]['input_schema']
    assert changed[0]['search_text'] != cards[0]['search_text']


def test_capability_equivalence_and_full_task_coverage_are_separate():
    caps = [{'cap_id': 'a', 'acceptable': ['s/a', 's/equivalent']}, {'cap_id': 'b', 'acceptable': ['s/b']}]
    r = cutoff_evidence(['s/equivalent', 's/other', 's/b'], caps, 2, ['s/a', 's/b'], {'s/equivalent'})
    assert r['found_cap_ids'] == ['a'] and not r['complete'] and r['complete_before']
    assert r['lost_cap_ids'] == r['collateral_lost_cap_ids'] == ['b']
    assert cutoff_evidence(['s/equivalent', 's/b'], caps, 2, [], set())['complete']


def test_retriever_ties_remain_stable_and_cutoff_is_not_rank_direction():
    assert ordered(np.array([[0, 1, 1, 0]])).tolist() == [[1, 2, 0, 3]]
    r = cutoff_evidence(['x'] * 5 + ['s/a'], [{'cap_id': 'a', 'acceptable': ['s/a']}], 5, [], {'s/a'})
    assert r['found_capabilities'] == 0


def test_aggregation_keeps_deployment_scenarios_separate_and_counts_losses():
    cap = [{'cap_id': 'a', 'acceptable': ['s/a']}]
    lost = cutoff_evidence(['s/x'], cap, 1, ['s/a'], {'s/x'})
    won = cutoff_evidence(['s/a'], cap, 1, ['s/x'], {'s/a'})
    rows = [{'scenario_id': scenario, 'strategy': 'bm25', 'mode': 'single', 'condition': 'structured',
             'tool_id': tool, 'cutoffs': {k: {**v, 'context_tokens': 10, 'description_characters': 20} for k in ('5', '10')}}
            for scenario, tool, v in [('one', 's/x', lost), ('two', 's/a', won)]]
    data = aggregate(rows)
    assert len(data) == 4
    assert next(r for r in data if r['scenario_id'] == 'one')['lost_complete_tasks'] == 1
    assert next(r for r in data if r['scenario_id'] == 'two')['newly_complete_tasks'] == 1


def test_measure_saves_both_cutoffs_and_the_actual_context_cost():
    from toolscope.diagnosis.repair_study import measure
    cards = [card(f's/{i}') for i in range(6)]
    task = {'task_id': 't', 'question': 'find', 'capabilities': [{'cap_id': 'a', 'acceptable': ['s/5']}]}
    scenario = {'scenario_id': 'single', 'condition': 'translation', 'mode': 'single',
                'tool_id': 's/5', 'edited_ids': ['s/5']}
    result = measure(task, np.arange(6), np.arange(6)[::-1], cards, 'bm25', scenario,
                     ['s/5', 's/0'], ['s/5'], len)
    assert result['target_ranks'] == {'s/5': 6}
    assert not result['cutoffs']['5']['complete'] and result['cutoffs']['10']['complete']
    assert result['cutoffs']['5']['lost_cap_ids'] == ['a']
    assert result['tokens'] == result['cutoffs']['10']['context_tokens']
    assert result['tokens'] > result['cutoffs']['5']['context_tokens']


def test_rebuilt_bm25_identity_control_and_unknown_query_keep_original_order():
    from toolscope.diagnosis.repair_study import bm25_scores
    cards = edited_cards([card('s/a', 'weather'), card('s/b', 'food')], {'s/a': 'weather', 's/b': 'food'})
    original = bm25_scores(cards, ['weather', 'unseenwords'])
    unchanged = bm25_scores(edited_cards(cards, {'s/a': 'weather'}), ['weather', 'unseenwords'])
    assert np.array_equal(original, unchanged)
    assert ordered(original)[1].tolist() == [0, 1]

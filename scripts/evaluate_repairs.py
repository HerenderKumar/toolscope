"""D-080: evaluate frozen single and joint description edits on the full task-question set.

Run: HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/evaluate_repairs.py
Original indexes, old results and task outcomes are read-only. No generation or tool execution.
"""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np

from toolscope.diagnosis.repair_controls import sha
from toolscope.diagnosis.repair_study import aggregate, bm25_scores, edited_cards, measure, ordered
from toolscope.log import get_logger
from toolscope.retrievers.base import CATALOG, INDEX_ROOT, dense_doc_text, load_cards

OUT = Path('results/repair_study_v1')
log = get_logger('toolscope.repair_study')


def read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def render_report(data):
    lines = ['# Description repair study v1 (D-080)', '',
             'Exploratory, selected-tool case study on original task questions. Discovery is measured; task success is not.',
             'Translations use local Qwen plus assistant corrections; structured texts reuse DeepSeek D-073 outputs. '
             'Their difference cannot isolate a prompt/model effect. No held-out generalization or significance claim.', '',
             '## Joint deployment: each complete edit set measured together', '',
             '| Condition / edit set | Retriever | k | Capabilities / 255 | Complete tasks / 95 | Gained / lost capabilities | Newly / lost complete tasks | Mean context tokens |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in data['joint']:
        label = 'structured (same 12 tools)' if r['mode'] == 'joint_chinese' else r['condition']
        lines.append(f"| {label} | {r['strategy']} | {r['k']} | {r['capabilities_found']} | {r['complete_tasks']} | "
                     f"{r['gained_cap_ids']} / {r['lost_cap_ids']} | {r['newly_complete_tasks']} / {r['lost_complete_tasks']} | {r['context_tokens_mean']:.1f} |")
    lines += ['', 'Capabilities use frozen acceptable-tool equivalence classes. Complete-task discovery requires every capability.',
              'Context uses the existing Qwen2.5-3B tokenizer and schema serialization, including unchanged input schemas.',
              'The translation set changes 12 descriptions; the full structured set contains 43. The matched structured set edits the same 12 tools as translation. English translation controls are identity edits.', '',
              '## Historical candidate probes (separate denominator)', '',
              '| Retriever | Improved | Unchanged | Worsened | Candidate-capability rescues @5 | Rescues @10 |',
              '|---|---:|---:|---:|---:|---:|']
    for strategy, r in data['historical_candidates'].items():
        lines.append(f"| {strategy} | {r['improved']} | {r['unchanged']} | {r['worsened']} | {r['rescues_at_5']} / 95 | {r['rescues_at_10']} / 95 |")
    lines += ['', 'The historical 110 rows share tasks/tools. A rescued capability is not a successful task. '
              'Do not add separate single-edit gains to estimate the joint catalog.', '',
              '## Four-cell repair choice example (historical wuwa probe)', '',
              '| Description | BM25 rank | Semantic rank |', '|---|---:|---:|']
    for condition, ranks in data['example']['ranks'].items():
        lines.append(f"| {condition} | {ranks['bm25']} | {ranks['semantic']} |")
    lines += ['', f"Tool: `{data['example']['tool_id']}`; task: `{data['example']['task_id']}`. "
              'tradeoffs.jsonl holds every selected-tool/task combination, including failures and regressions.', '',
              '## Faithfulness and limits', '',
              f"Assistant card review flagged concerns in {data['structured_review_concerns']} of 43 historical rewrites. "
              'The original texts remain frozen for measurement; this is not approval to deploy them.',
              'Six of 12 raw local translations needed corrections before scoring. Raw generations and all corrections are saved.',
              'No author/human annotation or live execution validation was performed in this review.',
              'Per-tool length and per-query context costs, complete-task losses and collateral capability losses are exported.',
              'Actual agent-query repair, multilingual retrieval and stochastic end-to-end agent reruns remain untested.', '',
              f"Controls: {data['baseline_comparisons']} saved-baseline comparisons; {data['noop_comparisons']} unchanged-description query/strategy comparisons; zero mismatches."]
    return '\n'.join(lines) + '\n'


def main():
    import torch  # before FAISS on macOS
    from transformers import AutoTokenizer
    from toolscope.retrievers.registry import MODEL_CONFIG, MODELS, encoder

    torch.set_num_threads(4)
    frozen = json.loads((OUT / 'conditions_manifest.json').read_text())
    if frozen['conditions_sha256'] != sha((OUT / 'conditions.jsonl').read_text()):
        raise ValueError('Frozen conditions changed')
    if frozen['review_sha256'] != sha((OUT / 'faithfulness_review.json').read_text()):
        raise ValueError('Frozen review changed')
    if frozen['raw_sha256'] != sha((OUT / 'translation_raw.jsonl').read_text()):
        raise ValueError('Frozen raw translation provenance changed')
    for path, h in frozen['source_hash'].items():
        if sha(Path(path).read_text()) != h:
            raise ValueError(f'Frozen source changed: {path}')
    conditions = read(OUT / 'conditions.jsonl')
    cards = load_cards(); ids = [c['tool_id'] for c in cards]; index = {t: i for i, t in enumerate(ids)}
    if json.loads((INDEX_ROOT / 'semantic/ids.json').read_text()) != ids:
        raise ValueError('Semantic vector order differs from catalog')
    tasks = read(Path('data/gold/gold.jsonl'))
    queries = [t['question'] for t in tasks]
    enc = encoder('semantic')
    qv = enc.encode([MODEL_CONFIG['semantic']['query_prefix'] + q for q in queries])
    vectors = np.load(INDEX_ROOT / 'semantic/vectors.npy')
    scores = {'bm25': bm25_scores(cards, queries), 'semantic': qv @ vectors.T}
    orders = {s: ordered(v) for s, v in scores.items()}
    baseline = {(t['task_id'], s): [ids[i] for i in orders[s][j]] for j, t in enumerate(tasks) for s in scores}
    saved = {(r['task_id'], r['strategy']): [t for t, _ in r['top_k'][:20]] for r in read(Path('logs/retrieval_k20.jsonl'))
             if r['strategy'] in scores}
    for key, ranked in baseline.items():
        if saved[key] != ranked[:20]:
            raise ValueError(f'Saved baseline mismatch: {key}')
    targets = [r['tool_id'] for r in conditions]
    no_cards = edited_cards(cards, {r['tool_id']: r['original'] for r in conditions})
    no_vectors = vectors.copy()
    no_vectors[[index[t] for t in targets]] = enc.encode([dense_doc_text(no_cards[index[t]]) for t in targets])
    no_scores = {'bm25': bm25_scores(no_cards, queries), 'semantic': qv @ no_vectors.T}
    for s in scores:
        if not np.array_equal(ordered(no_scores[s]), orders[s]):
            raise ValueError(f'Unchanged-description full ranking mismatch: {s}')
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
    count_tokens = lambda text: len(tok.encode(text))
    generated = [(r['tool_id'], cond, r[cond]) for r in conditions for cond in ('translation', 'structured') if r[cond] != r['original']]
    docs = [dense_doc_text(edited_cards([cards[index[t]]], {t: text})[0]) for t, _, text in generated]
    edited_vectors = {(t, c): v for (t, c, _), v in zip(generated, enc.encode(docs))}
    scenarios = [{'scenario_id': 'original', 'condition': 'original', 'mode': 'control', 'tool_id': None, 'edited_ids': []},
                 {'scenario_id': 'unchanged', 'condition': 'unchanged', 'mode': 'control', 'tool_id': None, 'edited_ids': []}]
    for condition in ('translation', 'structured'):
        scenarios += [{'scenario_id': f'{condition}:{r["tool_id"]}', 'condition': condition, 'mode': 'single',
                       'tool_id': r['tool_id'], 'edited_ids': [r['tool_id']] if r[condition] != r['original'] else []} for r in conditions]
        scenarios.append({'scenario_id': f'{condition}:joint', 'condition': condition, 'mode': 'joint', 'tool_id': None,
                          'edited_ids': [r['tool_id'] for r in conditions if r[condition] != r['original']]})
    scenarios.append({'scenario_id': 'structured:chinese_joint', 'condition': 'structured', 'mode': 'joint_chinese',
                      'tool_id': None, 'edited_ids': [r['tool_id'] for r in conditions if r['translation_applied']]})
    by_id = {r['tool_id']: r for r in conditions}
    measurements = []
    with (OUT / 'measurements.jsonl').open('w') as output:
        for number, scenario in enumerate(scenarios, 1):
            condition = scenario['condition']
            edits = {t: by_id[t][condition] for t in scenario['edited_ids']}
            changed_cards = edited_cards(cards, edits)
            if not edits:
                current = no_scores if condition == 'unchanged' else scores
            else:
                current_vectors = vectors.copy()
                for t in edits:
                    current_vectors[index[t]] = edited_vectors[(t, condition)]
                current = {'bm25': bm25_scores(changed_cards, queries), 'semantic': qv @ current_vectors.T}
            selected_targets = [scenario['tool_id']] if scenario['mode'] == 'single' else targets
            for strategy, matrix in current.items():
                order = ordered(matrix)
                for j, task in enumerate(tasks):
                    row = measure(task, order[j], matrix[j], changed_cards, strategy, scenario,
                                  baseline[(task['task_id'], strategy)], selected_targets, count_tokens)
                    measurements.append(row)
                    output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
            if number % 10 == 0:
                log.info('repair study progress', extra={'ctx': {'scenarios': number, 'total': len(scenarios)}})
    lookup = {(r['scenario_id'], r['task_id'], r['strategy']): r for r in measurements}
    historical = read(Path('results/whatif.jsonl'))
    historical_summary = {}
    for strategy in scores:
        for r in historical:
            before = lookup[('original', r['task_id'], strategy)]['target_ranks'][r['tool_id']]
            after = lookup[(f'structured:{r["tool_id"]}', r['task_id'], strategy)]['target_ranks'][r['tool_id']]
            if [before, after] != r[strategy]:
                raise ValueError(f'Historical probe mismatch: {strategy}, {r["tool_id"]}, {r["task_id"]}')
        direction = Counter('improved' if r[strategy][1] < r[strategy][0] else 'worsened' if r[strategy][1] > r[strategy][0] else 'unchanged' for r in historical)
        historical_summary[strategy] = {**direction, **{f'rescues_at_{k}': len({(r['task_id'], r['cap_id']) for r in historical if r[strategy][0] > k >= r[strategy][1]}) for k in (5, 10)}}
    tradeoffs = []
    for r in conditions:
        for task in tasks:
            ranks = {c: {s: lookup[('original' if c == 'original' else f'{c}:{r["tool_id"]}', task['task_id'], s)]['target_ranks'][r['tool_id']] for s in scores} for c in ('original', 'translation', 'structured')}
            tradeoffs.append({'tool_id': r['tool_id'], 'task_id': task['task_id'], 'ranks': ranks,
                              'inside_cutoff': {str(k): {c: {s: n <= k for s, n in value.items()} for c, value in ranks.items()} for k in (5, 10)}})
    (OUT / 'tradeoffs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in tradeoffs))
    lengths = [{'tool_id': r['tool_id'], 'condition': c, 'characters': len(r[c]), 'tokens': count_tokens(r[c]),
                'translation_applied': r['translation_applied']} for r in conditions for c in ('original', 'translation', 'structured')]
    (OUT / 'description_lengths.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in lengths))
    summaries = aggregate(measurements)
    review = json.loads((OUT / 'faithfulness_review.json').read_text())
    example_task = next(r['task_id'] for r in historical if r['tool_id'] == 'wuwa-mcp/get_character_info')
    data = {'version': 'D-080', 'exploratory': True, 'query_kind': 'original_task_question', 'tasks': len(tasks),
            'tools': len(targets), 'translation_tools': frozen['translations'], 'scenarios': len(scenarios),
            'measurements': len(measurements), 'baseline_comparisons': len(saved), 'noop_comparisons': 2 * len(tasks),
            'historical_probe_comparisons': 2 * len(historical), 'mismatches': 0,
            'joint': [r for r in summaries if r['mode'].startswith('joint') or r['condition'] == 'original'],
            'all_scenarios': summaries, 'historical_candidates': historical_summary,
            'structured_review_concerns': sum(r['status'] == 'concern' for r in review['structured']),
            'example': next(r for r in tradeoffs if r['tool_id'] == 'wuwa-mcp/get_character_info' and r['task_id'] == example_task)}
    inputs = [CATALOG, Path('data/gold/gold.jsonl'), Path('results/rewrites.jsonl'), Path('results/whatif.jsonl'),
              Path('logs/retrieval_k20.jsonl'), OUT/'conditions.jsonl', OUT/'conditions_manifest.json', OUT/'faithfulness_review.json', OUT/'translation_raw.jsonl',
              Path(__file__), Path('src/toolscope/diagnosis/repair_study.py'), Path('src/toolscope/diagnosis/repair_controls.py'),
              Path('src/toolscope/retrievers/registry.py'), Path('src/toolscope/retrievers/runlog.py')]
    inputs += [p for p in INDEX_ROOT.rglob('*') if p.is_file()]
    data['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    data['model_config'] = {'semantic': MODELS['semantic'], 'semantic_config': MODEL_CONFIG['semantic'], 'device': 'cpu',
                            'tokenizer': 'Qwen/Qwen2.5-3B-Instruct'}
    (OUT / 'summary.json').write_text(json.dumps(data, indent=1) + '\n')
    (OUT / 'report.md').write_text(render_report(data))
    print(f'tools={len(targets)} tasks={len(tasks)} scenarios={len(scenarios)} measurements={len(measurements)} '
          f'baseline_checks={len(saved)} noop_checks={2*len(tasks)} historical_checks={2*len(historical)} mismatches=0 out={OUT}')


if __name__ == '__main__':
    main()

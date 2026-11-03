"""D-079: replay saved agent queries through six local CPU retrievers, without executing tools.

Run: HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/replay_agent_searches.py
Resume with the same command; cached successes are reused only for the same input/configuration fingerprint.
"""

import hashlib
import json
import os
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from toolscope.diagnosis.search_evidence import collect, query_id, digest, annotate_events  # noqa: E402
from toolscope.diagnosis.runs import RUNS  # noqa: E402
from toolscope.retrievers.base import STRATEGIES, INDEX_ROOT, CATALOG, load_cards, dense_doc_text  # noqa: E402
from toolscope.log import get_logger  # noqa: E402

OUT = Path('results/diagnosis_v3')
log = get_logger('toolscope.replay_agent_searches')


def replay_one(retriever, strategy, query, depth, n_tools, fingerprint, fallback=None):
    start = time.perf_counter()
    qid = query_id(query)
    rec = {'query_id': qid, 'task_id': None, 'strategy': strategy, 'query': query, 'k': depth,
           'depth': depth, 'gold_rank': None, 'verdict': None, 'tokens': None, 'fallback_used': False,
           'fingerprint': fingerprint, 'replay_id': digest([fingerprint, strategy, qid])}
    try:
        full = retriever.retrieve(query, n_tools)
        if len(full) < depth or len({t for t, _ in full}) != len(full):
            raise ValueError('incomplete or duplicate retrieval ranking')
        rec.update(status='ok', top_k=full[:depth])
    except Exception as error:
        rec.update(status='error', error=str(error), top_k=[])
        log.exception('replay failed', extra={'ctx': {'strategy': strategy, 'query_id': qid}})
        if fallback is not None and strategy != 'bm25':
            try:
                rec.update(fallback_used=True, fallback_strategy='bm25', fallback_top_k=fallback.retrieve(query, depth))
                log.warning('downgraded to BM25; requested-strategy replay remains an error',
                            extra={'ctx': {'strategy': strategy, 'query_id': qid}})
            except Exception:
                log.exception('BM25 fallback failed', extra={'ctx': {'query_id': qid}})
    rec['latency_ms'] = round((time.perf_counter() - start) * 1000, 3)
    return rec


def main():
    import torch  # before FAISS (documented macOS OpenMP requirement)
    from huggingface_hub import try_to_load_from_cache
    from toolscope.retrievers.registry import get_retriever, encoder, MODELS, MODEL_CONFIG, RERANK_TOP_N, RRF_K, TOP_SERVERS

    torch.set_num_threads(int(os.environ.get('TOOLSCOPE_REPLAY_THREADS', '4')))
    runs, events = collect()
    queries = list(dict.fromkeys(e['query'] for e in events if e['replay_eligible']))
    cards = load_cards(); n_tools = len(cards)
    depth = min(n_tools, max(20, max((e['k'] for e in events if e['replay_eligible']), default=0)))
    inputs = {Path(f) / 'agent.jsonl' for folders in RUNS.values() for f in folders if (Path(f) / 'agent.jsonl').exists()}
    inputs |= {p for p in INDEX_ROOT.rglob('*') if p.is_file()}
    inputs |= {p for p in Path('src/toolscope/retrievers').glob('*.py')}
    inputs |= {CATALOG, Path('src/toolscope/diagnosis/search_evidence.py'), Path('src/toolscope/diagnosis/runs.py'), Path(__file__)}
    snapshots = {name: str(try_to_load_from_cache(model, 'config.json')) for name, model in MODELS.items()}
    config = {'models': MODELS, 'model_config': MODEL_CONFIG, 'model_cache_configs': snapshots, 'device': 'cpu',
              'rerank_top_n': RERANK_TOP_N, 'rrf_k': RRF_K, 'top_servers': TOP_SERVERS, 'depth': depth,
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)}}
    fingerprint = digest(config)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'replay_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Replay inputs changed. Preserve this directory and use a new version before replaying.')
    manifest_path.write_text(json.dumps({'fingerprint': fingerprint, **config}, indent=1) + '\n')
    cache_path = OUT / 'replay.jsonl'
    replay = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            if r['fingerprint'] != fingerprint:
                raise ValueError('Incompatible replay cache')
            if r['status'] == 'ok':
                replay[(r['strategy'], r['query_id'])] = r
    retrievers = {}
    for name in STRATEGIES:
        try:
            if name == 'rerank':
                from sentence_transformers import CrossEncoder
                from toolscope.retrievers.rerank import RerankRetriever
                model = CrossEncoder(MODELS['rerank'], max_length=512, device='cpu')
                scorer = lambda q, docs: [float(s) for s in model.predict([(q, d) for d in docs], show_progress_bar=False)]
                retriever = RerankRetriever(get_retriever('hybrid'), scorer,
                                           {c['tool_id']: dense_doc_text(c) for c in cards}, RERANK_TOP_N, n_tools)
            else:
                retriever = get_retriever(name)
            retriever.retrieve = lru_cache(maxsize=8)(retriever.retrieve)
            retrievers[name] = retriever
        except Exception as error:
            log.exception('model unavailable', extra={'ctx': {'strategy': name}})
            class FailedRetriever:
                def __init__(self, message): self.message = message
                def retrieve(self, query, k): raise RuntimeError(self.message)
            retrievers[name] = FailedRetriever(str(error))
    for name in ('semantic', 'toolret'):
        try:
            enc = encoder(name)
            original = enc.encode
            cached = lru_cache(maxsize=8)(lambda texts, fn=original: fn(list(texts)))
            enc.encode = lambda texts, fn=cached: fn(tuple(texts))
        except Exception:
            pass  # model-load failure already recorded; replay_one records each missing measurement
    written = 0
    log.info('replay ready', extra={'ctx': {'queries': len(queries), 'events': len(events), 'cached': len(replay)}})
    with cache_path.open('a') as output:
        for i, query in enumerate(queries, 1):
            for name in STRATEGIES:
                key = (name, query_id(query))
                if key in replay:
                    continue
                rec = replay_one(retrievers[name], name, query, depth, n_tools, fingerprint, retrievers['bm25'])
                replay[key] = rec
                output.write(json.dumps(rec, ensure_ascii=False) + '\n'); output.flush()
                written += 1
            if i % 50 == 0:
                log.info('replay progress', extra={'ctx': {'queries_done': i, 'queries_total': len(queries)}})
    annotated = annotate_events(events, replay)
    (OUT / 'search_events.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in annotated))
    summary = {'version': 'D-079', 'trajectories': sum(map(len, runs.values())), 'search_events': len(events),
               'unique_queries': len(queries), 'measurements': len(replay),
               'errors': sum(r['status'] != 'ok' for r in replay.values()),
               'original_strategy_matches': sum(e['original_strategy_reproduced'] is True for e in annotated),
               'original_strategy_mismatches': sum(e['original_strategy_reproduced'] is False for e in annotated),
               'original_strategy_unavailable': sum(e['original_strategy_reproduced'] is None for e in annotated),
               'cutoffs': dict(Counter(str(e['k']) for e in events)),
               'truncated_search_results': sum(bool(e['search_result_truncated']) for e in events),
               'fingerprint': fingerprint}
    (OUT / 'replay_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(f"events={len(events)} unique_queries={len(queries)} measurements={len(replay)} new={written} "
          f"errors={summary['errors']} matches={summary['original_strategy_matches']} out={OUT}")


if __name__ == '__main__':
    main()

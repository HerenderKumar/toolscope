"""Generate local translation controls using tool descriptions only; never loads tasks or queries.

Run: PYTHONPATH=src .venv/bin/python scripts/freeze_repair_controls.py
Raw generations are cached; --freeze requires the separately reviewed text/notes file.
"""

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import httpx

from toolscope.diagnosis.repair_controls import SYSTEM, VERSION, condition_rows, needs_translation, sha, translation_messages
from toolscope.log import get_logger
from toolscope.retrievers.base import CATALOG, load_cards

OUT = Path('results/repair_study_v1')
MODEL = 'qwen2.5:7b-instruct'
log = get_logger('toolscope.repair_controls')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'conditions_manifest.json').exists():
        raise ValueError('Conditions already frozen; preserve them and use a new version for changes')
    rewrites = [json.loads(l) for l in Path('results/rewrites.jsonl').read_text().splitlines()]
    raw_path = OUT / 'translation_raw.jsonl'
    raw = {r['tool_id']: r for l in raw_path.read_text().splitlines() if (r := json.loads(l))} if raw_path.exists() else {}
    wanted = [r for r in rewrites if needs_translation(r['original'])]
    options = {'temperature': 0, 'seed': 42, 'num_gpu': 0, 'num_thread': 4, 'num_ctx': 2048, 'num_predict': 512}
    config = {'model': MODEL, 'prompt_version': VERSION, 'system': SYSTEM, 'options': options}
    source_hash = {str(p): sha(p.read_text()) for p in [CATALOG, Path('results/rewrites.jsonl')]}
    calls = 0
    if not args.freeze:
        base_url = os.environ.get('TOOLSCOPE_OLLAMA_URL', 'http://127.0.0.1:11434')
        parsed = urlparse(base_url)
        if parsed.hostname not in {'localhost', '127.0.0.1', '::1'} or parsed.scheme != 'http':
            raise ValueError('Translation generation is restricted to local Ollama')
        with httpx.Client(timeout=600) as client, raw_path.open('a') as output:
            tags = client.get(f'{base_url}/api/tags'); tags.raise_for_status()
            model_digest = next(m['digest'] for m in tags.json()['models'] if m['name'] == MODEL)
            for row in wanted:
                cached = raw.get(row['tool_id'])
                fingerprint = sha([config, model_digest, row['original']])
                if cached:
                    if cached['fingerprint'] != fingerprint:
                        raise ValueError('Translation cache inputs changed')
                    continue
                response = client.post(f'{base_url}/api/chat', json={
                    'model': MODEL, 'stream': False, 'keep_alive': '5m', 'options': options,
                    'messages': translation_messages(row['original'])})
                response.raise_for_status(); body = response.json()
                record = {'tool_id': row['tool_id'], 'original': row['original'], 'raw_translation': body['message']['content'].strip(),
                          'model_digest': model_digest, 'fingerprint': fingerprint, 'config': config,
                          'stop_reason': body.get('done_reason'), 'input_tokens': body.get('prompt_eval_count'),
                          'output_tokens': body.get('eval_count')}
                if record['stop_reason'] == 'length' or not record['raw_translation']:
                    raise ValueError('Incomplete translation output; preserve and review before retrying')
                output.write(json.dumps(record, ensure_ascii=False) + '\n'); output.flush()
                raw[row['tool_id']] = record; calls += 1
                log.info('translation saved', extra={'ctx': {'tool_id': row['tool_id'], 'done': len(raw), 'total': len(wanted)}})
        print(f'tools={len(rewrites)} translations={len(raw)} new_local_calls={calls} frozen=false out={OUT}')
        return
    review_path = OUT / 'faithfulness_review.json'
    review = json.loads(review_path.read_text())
    translations = {r['tool_id']: r['translation'] for r in review['translations']}
    if {r['tool_id'] for r in review['structured']} != {r['tool_id'] for r in rewrites}:
        raise ValueError('All structured rewrites require a faithfulness review')
    for r in review['translations']:
        if r['raw_sha256'] != sha(raw[r['tool_id']]['raw_translation']):
            raise ValueError('Translation review does not match raw generation')
    rows = condition_rows(load_cards(), rewrites, translations)
    target = OUT / 'conditions.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    manifest = {'version': 'D-080', 'scope': 'Tool-only conditions frozen before this study evaluation',
                'exploratory': True, 'config': config, 'tools': len(rows), 'translations': len(translations),
                'generation_provenance_differs': True, 'source_hash': source_hash,
                'conditions_sha256': sha(target.read_text()), 'review_sha256': sha(review_path.read_text()),
                'raw_sha256': sha(raw_path.read_text()), 'schema_changes': 0,
                'reviewer': review['reviewer'], 'created_at': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    (OUT / 'conditions_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'tools={len(rows)} translations={len(translations)} new_local_calls=0 frozen=true out={OUT}')


if __name__ == '__main__':
    main()

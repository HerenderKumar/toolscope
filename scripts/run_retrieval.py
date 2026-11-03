"""Run the 6 strategies over the 95 tasks and write logs/retrieval.jsonl (one line per query).

Each strategy ranks all tools; the log keeps the top-k and the best strict-gold rank (D-016).
Run:  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/run_retrieval.py
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from toolscope.log import get_logger
from toolscope.retrievers.base import STRATEGIES, load_cards
from toolscope.retrievers.registry import get_retriever
from toolscope.retrievers.runlog import make_record, schema_tokens

log = get_logger("toolscope.run_retrieval")

GOLD = Path(os.environ.get("TOOLSCOPE_GOLD", "data/gold/gold.jsonl"))
QRELS = Path(os.environ.get("TOOLSCOPE_QRELS", "data/gold/qrels_strict.json"))
OUT = Path(os.environ.get("TOOLSCOPE_RETRIEVAL_LOG", "logs/retrieval.jsonl"))
K = int(os.environ.get("TOOLSCOPE_K", "10"))
TOKENIZER = os.environ.get("TOOLSCOPE_TOKENIZER", "Qwen/Qwen2.5-3B-Instruct")


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    count = lambda text: len(tok.encode(text))
    cards = {c["tool_id"]: c for c in load_cards()}
    n_tools = len(cards)
    tasks = [json.loads(line) for line in GOLD.read_text().splitlines()]
    qrels = json.loads(QRELS.read_text())
    run_id = datetime.now(timezone.utc).strftime("retrieval-%Y%m%dT%H%M%SZ")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = errors = 0
    with OUT.open("w", encoding="utf-8") as f:
        for strategy in STRATEGIES:
            retriever = get_retriever(strategy)
            retriever.retrieve("warm-up query", 1)  # load models before timing
            for task in tasks:
                try:
                    start = time.perf_counter()
                    full = retriever.retrieve(task["question"], n_tools)
                    latency = (time.perf_counter() - start) * 1000
                    tokens = schema_tokens(cards, [t for t, _ in full[:K]], count)
                    rec = make_record(run_id, strategy, task["task_id"], K, full,
                                      qrels[task["task_id"]], latency, tokens)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                except Exception:
                    errors += 1
                    log.exception("retrieval failed", extra={"ctx": {"strategy": strategy,
                                                                    "task_id": task["task_id"]}})
            log.info("strategy done", extra={"ctx": {"strategy": strategy}})
    print(f"strategies={len(STRATEGIES)} tasks={len(tasks)} lines={written} errors={errors} "
          f"run_id={run_id} out={OUT}")


if __name__ == "__main__":
    main()

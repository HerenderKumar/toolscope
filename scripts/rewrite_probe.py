"""Phase 5 Steps 2-3: rewrite the descriptions of the tools that no strategy finds (query-blind, DeepSeek V3.2),
then measure each rewrite's rank effect on the task questions with the what-if editor.

Candidates = capabilities whose acceptable tools are in no strategy's top-10 (results/failure_labels.jsonl,
retrieval rows that are neither found nor found_by another strategy). Rewrites are cached in
results/rewrites.jsonl (re-runs cost nothing); rank effects go to results/whatif.jsonl. A capability counts as a
description fault (D-010 rule 5) when an acceptable tool's new rank is <= 10 under bm25 or semantic.

Run:  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/rewrite_probe.py
"""

import json
import os
from pathlib import Path

from toolscope.agent.backends import openai_chat
from toolscope.diagnosis.rewrite import PROMPT_VERSION, rewrite_messages
from toolscope.diagnosis.whatif import WhatIf
from toolscope.gateway.server import load_env_file
from toolscope.log import get_logger
from toolscope.retrievers.base import load_cards

log = get_logger("toolscope.rewrite_probe")
MODEL = os.environ.get("TOOLSCOPE_REWRITE_MODEL", "deepseek/deepseek-v3.2")
GOLD = Path("data/gold/gold.jsonl")
LABELS = Path("results/failure_labels.jsonl")
REWRITES = Path("results/rewrites.jsonl")
WHATIF = Path("results/whatif.jsonl")
TOP = 10


def candidates(gold: dict[str, dict]) -> list[tuple[str, dict]]:
    """(task_id, capability) pairs that every strategy misses in its top-10."""
    keys = set()
    for line in LABELS.read_text().splitlines():
        r = json.loads(line)
        if r["scope"] == "retrieval" and r["label"] != "found" and not r["reason"].startswith("found_by"):
            keys.add((r["task_id"], r["cap_id"]))
    out = []
    for task_id, cap_id in sorted(keys):
        cap = next(c for c in gold[task_id]["capabilities"] if c["cap_id"] == cap_id)
        out.append((task_id, cap))
    return out


def main() -> None:
    load_env_file(Path(os.environ.get("TOOLSCOPE_ENV_FILE", ".env")))
    gold = {json.loads(l)["task_id"]: json.loads(l) for l in GOLD.read_text().splitlines()}
    cards = {c["tool_id"]: c for c in load_cards()}
    cands = candidates(gold)
    tools = sorted({t for _, cap in cands for t in cap["acceptable"]})

    cache = {}
    if REWRITES.exists():
        for line in REWRITES.read_text().splitlines():
            r = json.loads(line)
            if r["model"] == MODEL and r["prompt_version"] == PROMPT_VERSION:
                cache[r["tool_id"]] = r
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
    chat = openai_chat(MODEL, usage, base_url=os.environ["TOOLSCOPE_API_BASE_URL"],
                       api_key=os.environ["TOOLSCOPE_API_KEY"], temperature=0.0)
    with REWRITES.open("a") as f:
        for tool_id in tools:
            if tool_id in cache:
                continue
            try:
                text = chat(rewrite_messages(cards[tool_id]), None)["content"].strip()
            except Exception as e:  # one failed rewrite must not stop the probe
                log.error("rewrite failed", extra={"tool_id": tool_id, "error": str(e)[:300]})
                continue
            rec = {"tool_id": tool_id, "model": MODEL, "prompt_version": PROMPT_VERSION,
                   "original": cards[tool_id]["description"], "rewritten": text}
            cache[tool_id] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    editor = WhatIf(list(cards.values()))
    rows, fixed = [], 0
    for task_id, cap in cands:
        question = gold[task_id]["question"]
        best = None
        for tool_id in cap["acceptable"]:
            if tool_id not in cache:
                continue
            res = editor.whatif(tool_id, cache[tool_id]["rewritten"], [question])[0]
            row = {"task_id": task_id, "cap_id": cap["cap_id"], "tool_id": tool_id,
                   "bm25": res["bm25"], "semantic": res["semantic"]}
            row["moved_into_top10"] = [s for s in ("bm25", "semantic") if row[s][1] <= TOP]
            rows.append(row)
            if row["moved_into_top10"] and best is None:
                best = row
        fixed += best is not None
    WHATIF.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"candidates={len(cands)} tools={len(tools)} rewritten={len(cache)} new_llm_calls={usage['llm_calls']} "
          f"tokens_in={usage['input_tokens']} tokens_out={usage['output_tokens']} "
          f"description_faults={fixed} out={WHATIF}")


if __name__ == "__main__":
    main()

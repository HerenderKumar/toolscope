"""Phase 5 Step 4: write results/tool_health.jsonl (one row per catalog tool, most-missed first).

Run:  PYTHONPATH=src .venv/bin/python scripts/tool_health.py
"""

import json
from collections import Counter
from pathlib import Path

from toolscope.diagnosis.tool_health import tool_health
from toolscope.retrievers.base import load_cards

GOLD = Path("data/gold/gold.jsonl")
LABELS = Path("results/failure_labels.jsonl")
WHATIF = Path("results/whatif.jsonl")
OUT = Path("results/tool_health.jsonl")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def main() -> None:
    gold = {json.loads(l)["task_id"]: json.loads(l) for l in GOLD.read_text().splitlines()}
    rows = tool_health(load_cards(), gold, _jsonl(LABELS), _jsonl(WHATIF))
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    flags = Counter(f for r in rows for f in r["flags"])
    gold_tools = [r for r in rows if r["gold_caps"]]
    print(f"tools={len(rows)} gold_tools={len(gold_tools)} missed_at_least_once={sum(r['missed'] > 0 for r in rows)} "
          f"whatif_tools={sum(r['whatif'] is not None for r in rows)} flags={dict(sorted(flags.items()))} out={OUT}")


if __name__ == "__main__":
    main()

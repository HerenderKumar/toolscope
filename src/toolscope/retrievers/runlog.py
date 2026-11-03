"""One structured log line per query, in the fixed 10-field schema (+ ts, run_id)."""

import json
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from toolscope.retrievers.base import Ranking

LOG_FIELDS = ["query_id", "task_id", "strategy", "k", "top_k", "gold_rank",
              "verdict", "latency_ms", "tokens", "fallback_used"]


def gold_rank(ranked_ids: list[str], gold: Iterable[str]) -> int | None:
    """Best 1-based rank of any gold tool, or None if none is ranked."""
    gold = set(gold)
    return next((i for i, t in enumerate(ranked_ids, start=1) if t in gold), None)


def schema_tokens(cards: dict[str, dict[str, Any]], top_ids: list[str],
                  count: Callable[[str], int]) -> int:
    """Context cost: tokens of the top-k tool schemas an agent would have to read."""
    schemas = [{"name": cards[t]["tool_name"], "description": cards[t]["description"],
                "inputSchema": cards[t]["input_schema"]} for t in top_ids]
    return count(json.dumps(schemas, ensure_ascii=False))


def make_record(run_id: str, strategy: str, task_id: str | None, k: int, full: Ranking,
                gold: Iterable[str], latency_ms: float, tokens: int,
                fallback_used: bool = False, query_id: str | None = None) -> dict[str, Any]:
    return {
        "query_id": query_id or f"{run_id}/{strategy}/{task_id}",
        "task_id": task_id,
        "strategy": strategy,
        "k": k,
        "top_k": [[t, round(s, 6)] for t, s in full[:k]],
        "gold_rank": gold_rank([t for t, _ in full], gold),
        "verdict": None,
        "latency_ms": round(latency_ms, 1),
        "tokens": tokens,
        "fallback_used": fallback_used,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
    }

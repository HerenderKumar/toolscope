"""The final D-063 agent runs: which log folders make up each (agent, strategy) run, and the latest record per task."""

import json
from pathlib import Path

# (agent, strategy) -> folders, in the D-063 "final data rule" order (fix folders last; later records win)
RUNS = {
    ("Qwen2.5-7B (local)", "bm25"): ["logs/7b_d063", "logs/7b_d063_fix"],
    ("Qwen2.5-7B (local)", "semantic"): ["logs/7b_d063_semantic", "logs/7b_d063_fix"],
    ("Qwen3-235B", "bm25"): ["logs/qwen235b/d063_bm25", "logs/qwen235b/d063_fix"],
    ("Qwen3-235B", "semantic"): ["logs/qwen235b/d063_semantic", "logs/qwen235b/d063_fix"],
    ("Sonnet 5.5", "bm25"): ["logs/sonnet/d063_old", "logs/sonnet", "logs/sonnet/d063_fix"],
    ("Kimi K2.6", "bm25"): ["logs/kimi/test5", "logs/kimi/d063_bm25_a", "logs/kimi/d063_bm25_b"],
    ("Kimi K2.6", "semantic"): ["logs/kimi/d063_semantic_a", "logs/kimi/d063_semantic_b"],
}


def load(agent: str, strategy: str, folders: list[str], root: Path = Path(".")) -> dict[str, dict]:
    """Latest record per task of one run (later folders and later lines win)."""
    best: dict[str, dict] = {}
    for folder in folders:
        path = root / folder / "agent.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("strategy") != strategy:
                continue
            # logs/sonnet/agent.jsonl also holds the pre-D-063 4,000-char runs; keep only D-063 records there
            if folder == "logs/sonnet" and "max_tool_result_chars" not in rec:
                continue
            best[rec["task_id"]] = rec
    return best

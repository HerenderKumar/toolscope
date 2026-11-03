"""Compact status of the agent runs, computed from the logs (cheap to read in a fresh session).

Run:  PYTHONPATH=src .venv/bin/python -m toolscope.agent.status
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from toolscope.agent.watch import SOFT_PATTERNS, classify_call, classify_error

AGENT_LOG = Path("logs/agent.jsonl")
RERUN_LOG = Path("logs/rerun/agent.jsonl")
TOTAL_TASKS = 95


def latest(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """One record per (strategy, task): the last one wins (re-run rule, D-028)."""
    out = {}
    for r in records:
        out[(r["strategy"], r["task_id"])] = r
    return out


def rerun_list(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """(strategy, task_id) whose latest record has an environment-classified tool error."""
    out = []
    for key, r in latest(records).items():
        calls = [s for s in r.get("steps", []) if s.get("type") == "call"]
        errors = [(s["tool_id"], s["error"]) for s in calls if s.get("error")]
        # temporary outages can also hide inside a "successful" result (e.g. arXiv "HTTP 503")
        errors += [(s["tool_id"], s.get("result_snippet") or "") for s in calls
                   if s.get("ok") and SOFT_PATTERNS.search((s.get("result_snippet") or "")[:400])]
        if any(classify_call(t, e) == "ENV" for t, e in errors):  # tool-aware: blocked/unreachable tools are EXT
            out.append(key)
    return sorted(out)


FORCE_RERUN = Path("logs/rerun/force_rerun.txt")


def forced_reruns(path: Path = FORCE_RERUN) -> list[tuple[str, str]]:
    """Extra (strategy, task_id) to re-run, one 'strategy task_id' per line ('#' = comment): tasks hit by an
    environment fault the error rules cannot see, e.g. a broken server returning errors inside results (D-034)."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        parts = line.split("#")[0].split()
        if len(parts) == 2:
            out.append((parts[0], parts[1]))
    return out


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def main() -> None:
    records = read(AGENT_LOG) + read(RERUN_LOG)
    best = latest(records)
    per_strategy = Counter(s for s, _ in best)
    print("progress:", ", ".join(f"{s} {n}/{TOTAL_TASKS}" for s, n in sorted(per_strategy.items())) or "no tasks yet")
    if records:
        last = max(records, key=lambda r: r.get("ts") or "")
        print("last task logged:", last.get("ts"), last["strategy"], last["task_id"][:8])
    reruns = sorted(set(rerun_list(records)) | set(forced_reruns()))
    print(f"rerun_list ({len(reruns)}):", ", ".join(f"{s}:{t[:8]}" for s, t in reruns) or "none")


if __name__ == "__main__":
    main()

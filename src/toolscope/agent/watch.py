"""Live watcher for agent runs: one summary line per finished task, with each tool error sorted into
ENV (environment/network: fix and re-run the task), AGENT (a genuine agent mistake: keep) or CHECK (review).

Run:  PYTHONPATH=src .venv/bin/python -m toolscope.agent.watch logs/agent.jsonl
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ENV_PATTERNS = re.compile(
    r"executable doesn't exist|failed to initialize|econnrefused|enotfound|eai_again|econnreset|etimedout|"
    r"connection (closed|refused|reset)|server disconnected|timed? ?out|no such module|modulenotfounderror|"
    r"no module named|permission denied|\b(403|406|429|5\d\d)\b|not acceptable|too many requests|rate limit|ssl|certificate|"
    r"network is unreachable|name or service not known|temporary failure|no launch config|"
    r"retrieval incomplete|incompleteread", re.I)  # download cut off mid-transfer (arXiv, 31 Oct)
AGENT_PATTERNS = re.compile(r"^invalid arguments:|^unknown tool |"
                            r"waitforselector: timeout|waiting for selector .* failed", re.I)  # selector absent (D-049)
# permanent external changes: re-running cannot help, and they are not the agent's or retriever's fault
# (live server rejects catalog-valid arguments = schema drift; DuckDuckGo anti-bot; retired APIs; D-033)
EXT_PATTERNS = re.compile(
    r"mcp error -32602|detected an anomaly|requests too quickly|status:? ?410\b|\b410 gone|was retired|"
    r"downstreamerror: unknown tool: |"  # server lists a tool it never implemented (doc-forge excel_read)
    r"unreachable from the evaluation network|"
    r"cannot query field|"  # upstream GraphQL schema changed (LeetCode get-user-profile)
    r"page structure may have changed|"  # site changed its HTML, scraper broken (Airbnb airbnb_search, D-044)
    r"data\.gov api error: request failed with status code 404|"  # CKAN API at catalog.data.gov retired (D-048)
    r"call tool generate_\w*_map failed: cannot read properties of undefined|"  # AntV remote map service (D-051)
    r"\"charcount\": 0\b|"  # YouTube returns empty captions to the transcript scraper (D-057)
    
    r"\btool '?[\w./-]+'? not found", re.I)
# a call that "succeeded" but whose result carries an error (e.g. Wikipedia's blocked client, D-030)
# status codes only count next to "status"/"HTTP" or as "(429, 'https…')", not as numbers inside data
SOFT_PATTERNS = re.compile(
    r"\berror\b(?!\"?\s*:\s*null)|failed to|\bexception\b|expecting value|"  # but not a JSON "error": null
    r"(status|http)\D{0,12}\b(403|404|406|410|429|5\d\d)\b|\((403|404|406|410|429|5\d\d), '?https?:|"
    r"\"charcount\": 0\b", re.I)  # an empty YouTube transcript (D-057)


# tools whose upstream source is unreachable from the evaluation network; their errors carry no usable text (D-041)
UNREACHABLE_TOOLS = {"trends-hub/get-weibo-trending", "trends-hub/get-netease-news-trending",
                     "trends-hub/get-weread-rank",  # weread.qq.com unreachable too (D-057)
                     "trends-hub/get-tencent-news-trending"}  # r.inews.qq.com unreachable (31 Oct, paid run)


# tools whose upstream blocks automated clients outright; every error is permanent, never worth a paid re-run
BLOCKED_TOOLS = {"duckduckgo-search/duckduckgo_web_search"}  # "anomaly" and "Rate limit exceeded" alike (31 Oct)


def classify_call(tool_id: str, error: str) -> str:
    if tool_id in UNREACHABLE_TOOLS or tool_id in BLOCKED_TOOLS:
        return "EXT"
    return classify_error(error)


def classify_error(error: str) -> str:
    if AGENT_PATTERNS.search(error):
        return "AGENT"
    if EXT_PATTERNS.search(error):
        return "EXT"
    if ENV_PATTERNS.search(error):
        return "ENV"
    return "CHECK"


def summarize(rec: dict[str, Any], n: int, gold_caps: dict[str, list[dict[str, Any]]]) -> str:
    calls = [s for s in rec.get("steps", []) if s["type"] == "call"]
    acceptable = {t for c in gold_caps.get(rec["task_id"], []) for t in c["acceptable"]}
    gold = sum(c["tool_id"] in acceptable for c in calls)
    ok = sum(bool(c.get("ok")) for c in calls)
    flags: dict[str, list[str]] = {"ENV": [], "EXT": [], "CHECK": [], "SOFT": []}
    for c in calls:
        if c.get("error"):
            kind = classify_call(c["tool_id"], c["error"])
            if kind in flags:
                flags[kind].append(f"{c['tool_id']}: {c['error'][:90]}")
        elif c.get("ok") and SOFT_PATTERNS.search((c.get("result_snippet") or "")[:400]):
            flags["SOFT"].append(f"{c['tool_id']}: {(c.get('result_snippet') or '')[:90]}")
    secs = (rec.get("latency_ms") or 0) / 1000
    line = (f"[{n}] {rec.get('strategy')} {rec['task_id'][:8]} {secs:.0f}s stop={str(rec.get('stop_reason'))[:30]} "
            f"calls={len(calls)} gold={gold} ran_ok={ok} net_ok={rec.get('network_ok')}")
    for kind, items in flags.items():
        if items:
            line += f" | {kind}: " + " ; ".join(dict.fromkeys(items))
    return line


def new_lines(path: Path, seen: int) -> list[tuple[int, str]]:
    """(1-based line number, line) for every line after the first `seen` lines."""
    lines = path.read_text().splitlines() if path.exists() else []
    return [(i, line) for i, line in enumerate(lines[seen:], start=seen + 1)]


def should_report(line: str, n: int, quiet: bool, every: int = 10) -> bool:
    """Quiet mode: only flagged tasks (ENV/SOFT/CHECK) and a progress line every `every` tasks."""
    return not quiet or " | " in line or n % every == 0


def follow(path: Path, seen: int = 0, quiet: bool = False, gold_path: Path = Path("data/gold/gold.jsonl"),
           poll_s: float = 2.0) -> None:
    gold_caps = {g["task_id"]: g["capabilities"] for g in map(json.loads, gold_path.read_text().splitlines())}
    while True:
        for n, line in new_lines(path, seen):
            seen = n
            out = summarize(json.loads(line), n, gold_caps)
            if should_report(out, n, quiet):
                print(out, flush=True)
        time.sleep(poll_s)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--quiet"]
    follow(Path(args[0] if args else "logs/agent.jsonl"), seen=int(args[1]) if len(args) > 1 else 0,
           quiet="--quiet" in sys.argv)

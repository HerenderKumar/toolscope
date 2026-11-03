"""Deterministic failure labels per (agent run, task, capability), precedence D-010. Reads only logs and the frozen
answer key; nothing here calls a model. The what-if step (bad description) is applied later by upgrading the
`pending_whatif` retrieval misses.

Offline lists: {strategy: {task_id: [tool_id, ...]}} ranked tool lists for the task question (top-20 when
available, so ranks 11-20 can be seen; rules use the first 10 for "found").
"""

import re
from typing import Any

from toolscope.agent.watch import BLOCKED_TOOLS, UNREACHABLE_TOOLS, classify_call

ENVIRONMENT = "environment failure"
SELECTION = "agent selection error"
COMPOSITION = "composition error"
RANKING = "ranking confusion"
DESCRIPTION = "bad description"
RETRIEVAL = "retrieval miss"
LABELS = (ENVIRONMENT, SELECTION, COMPOSITION, RANKING, DESCRIPTION, RETRIEVAL)
TOP = 10
NEAR = 20
EMPTY_TRANSCRIPT = re.compile(r'"charcount":\s*0\b', re.I)  # D-057: an OK call with no transcript
# broken upstream but silent (an OK call with an empty result) or never called by any agent, so the logs alone
# cannot show it: KRX data needs a login (D-046, D-057 load_all_tickers); OP.GG serves none of its tools (D-057)
KNOWN_BROKEN = {"kospi-kosdaq/get_stock_fundamental", "kospi-kosdaq/get_stock_market_cap",
                "kospi-kosdaq/get_stock_trading_volume", "kospi-kosdaq/load_all_tickers",
                "opgg-mcp/valorant-characters-statistics"}


def call_outcome(call: dict[str, Any]) -> str:
    """OK, or the failure class (EXT / ENV / AGENT / CHECK) of one logged call."""
    if call.get("ok"):
        if call["tool_id"] in KNOWN_BROKEN or EMPTY_TRANSCRIPT.search(call.get("result_snippet") or ""):
            return "EXT"
        return "OK"
    return classify_call(call["tool_id"], call.get("error") or "")


def broken_tools(records: list[dict[str, Any]]) -> set[str]:
    """Tools that never succeeded in any run and failed for an outside reason, plus the D-041 unreachable /
    blocked lists and the silent breaks in KNOWN_BROKEN."""
    ok, ext = set(), set()
    for rec in records:
        for step in rec["steps"]:
            if step.get("type") != "call":
                continue
            outcome = call_outcome(step)
            (ok if outcome == "OK" else ext if outcome == "EXT" else set()).add(step["tool_id"])
    return (ext - ok) | UNREACHABLE_TOOLS | BLOCKED_TOOLS | KNOWN_BROKEN


def _rank(tools: list[str], acceptable: set[str]) -> int | None:
    return next((i for i, t in enumerate(tools, 1) if t in acceptable), None)


def retrieval_label(cap: dict[str, Any], task_id: str, strategy: str,
                    offline: dict[str, dict[str, list[str]]]) -> tuple[str, str] | None:
    """Rules 4-6 for one strategy on the task question; None when this strategy's top-10 has the tool."""
    acceptable = set(cap["acceptable"])
    here = _rank(offline[strategy].get(task_id, []), acceptable)
    if here is not None and here <= TOP:
        return None
    others = [s for s in offline if s != strategy and (r := _rank(offline[s].get(task_id, []), acceptable))
              and r <= TOP]
    if others:
        return RANKING, "found_by:" + ",".join(sorted(others))
    if here is not None and here <= NEAR:
        return RANKING, f"rank_{here}"
    return RETRIEVAL, "pending_whatif"


def label_capability(cap: dict[str, Any], rec: dict[str, Any], strategy: str, solved_caps: set[str],
                     n_caps: int, broken: set[str], offline: dict[str, dict[str, list[str]]]) -> dict[str, str]:
    """Label for one capability of one run's task record (D-010). Returns {label, reason}; label 'solved' when
    a successful call to an acceptable tool exists."""
    acceptable = set(cap["acceptable"])
    calls = [s for s in rec["steps"] if s.get("type") == "call"]
    gold_calls = [(c, call_outcome(c)) for c in calls if c["tool_id"] in acceptable]
    if any(o == "OK" for _, o in gold_calls):
        return {"label": "solved", "reason": ""}
    # 1 environment
    if acceptable <= broken:
        return {"label": ENVIRONMENT, "reason": "all_acceptable_broken"}
    if any(o in ("EXT", "ENV") for _, o in gold_calls):
        return {"label": ENVIRONMENT, "reason": "gold_call_failed_outside"}
    # 2 agent selection: shortlisted but not used successfully
    shortlisted = {t for s in rec["steps"] for t in (s.get("shortlist") or s.get("shortlist_before") or [])}
    if acceptable & shortlisted:
        return {"label": SELECTION, "reason": "bad_call" if gold_calls else "not_called"}
    # 3 composition, or the agent's own queries missed what the task question finds
    ranked = offline.get(strategy, {}).get(rec["task_id"], [])
    in_top = (r := _rank(ranked, acceptable)) is not None and r <= TOP
    if in_top:
        other_solved = bool(solved_caps - {cap["cap_id"]})
        if n_caps > 1 and other_solved:
            return {"label": COMPOSITION, "reason": "dropped_step"}
        return {"label": SELECTION, "reason": "query"}
    # 4-6 retrieval side
    label, reason = retrieval_label(cap, rec["task_id"], strategy, offline)
    return {"label": label, "reason": reason}


def label_run(records: dict[str, dict[str, Any]], gold: dict[str, dict[str, Any]], strategy: str,
              broken: set[str], offline: dict[str, dict[str, list[str]]]) -> list[dict[str, str]]:
    """One row per (task, capability) of a run."""
    rows = []
    for task_id, rec in sorted(records.items()):
        caps = gold[task_id]["capabilities"]
        solved = {c["cap_id"] for c in caps
                  if any(s.get("type") == "call" and s["tool_id"] in c["acceptable"] and call_outcome(s) == "OK"
                         for s in rec["steps"])}
        for cap in caps:
            out = label_capability(cap, rec, strategy, solved, len(caps), broken, offline)
            rows.append({"task_id": task_id, "cap_id": cap["cap_id"], **out})
    return rows


def label_offline(gold: dict[str, dict[str, Any]], offline: dict[str, dict[str, list[str]]]) -> list[dict[str, str]]:
    """Retrieval-only labels for every strategy (rules 4-6); 'found' when the strategy's top-10 has the tool."""
    rows = []
    for strategy in offline:
        for task_id, task in sorted(gold.items()):
            for cap in task["capabilities"]:
                out = retrieval_label(cap, task_id, strategy, offline)
                label, reason = out if out else ("found", "")
                rows.append({"strategy": strategy, "task_id": task_id, "cap_id": cap["cap_id"],
                             "label": label, "reason": reason})
    return rows

"""D-078: scope saved D-073 interventions to their tested retrieval context.

This module only interprets saved measurements. It neither rewrites descriptions
nor runs retrievers. Query metadata identifies the offline task-question probe;
it must never be treated as a replay of the agent's own search query.
"""

import hashlib
import json
from collections import defaultdict
from typing import Any

from toolscope.diagnosis.classify import DESCRIPTION, RETRIEVAL, TOP

VERSION = "D-078"
QUERY_KIND = "offline_task_question"
PROVENANCE = "Reconstructed from saved D-073 task IDs and the unique cached rewrite per tool; not replayed"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()


def normalize_probes(probes: list[dict[str, Any]], rewrites: list[dict[str, Any]],
                     gold: dict[str, dict[str, Any]], cards: dict[str, dict[str, Any]],
                     cutoff: int = TOP) -> list[dict[str, Any]]:
    """Link legacy probes conservatively; ambiguous/missing identities fail closed.

    Do not choose a rewrite among multiple cached versions after seeing ranks.
    All before/after observations remain separate, including failed repairs.
    """
    if cutoff != TOP:
        raise ValueError("D-078 classifier cutoff is fixed at 10")
    cache: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rewrite in rewrites:
        cache[rewrite["tool_id"]].append(rewrite)
    evidence, seen = [], set()
    for line, probe in enumerate(probes, 1):
        task = gold[probe["task_id"]]
        cap = next(c for c in task["capabilities"] if c["cap_id"] == probe["cap_id"])
        tool = probe["tool_id"]
        if tool not in cap["acceptable"]:
            raise ValueError(f"Probe tool is not acceptable for {cap['cap_id']}: {tool}")
        if len(cache[tool]) != 1:
            raise ValueError(f"Need exactly one historical rewrite for {tool}; cannot infer probe version")
        rewrite = cache[tool][0]
        if rewrite["original"] != cards[tool]["description"]:
            raise ValueError(f"Original description differs from the catalog: {tool}")
        intervention = {k: rewrite[k] for k in ("tool_id", "model", "prompt_version")}
        intervention.update(original_sha256=query_hash(rewrite["original"]),
                            rewritten_sha256=query_hash(rewrite["rewritten"]))
        intervention["intervention_id"] = digest(intervention)
        for strategy in ("bm25", "semantic"):
            if strategy not in probe:
                continue
            pair = probe[strategy]
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(type(rank) is not int or not 1 <= rank <= len(cards) for rank in pair)):
                raise ValueError(f"Invalid measured ranks at probe line {line}: {strategy}")
            identity = {"task_id": task["task_id"], "cap_id": cap["cap_id"], "strategy": strategy,
                        "query_kind": QUERY_KIND, "query_sha256": query_hash(task["question"]),
                        "cutoff": cutoff, "intervention_id": intervention["intervention_id"]}
            eid = digest(identity)
            if eid in seen:
                raise ValueError(f"Duplicate measured intervention: {eid}")
            seen.add(eid)
            evidence.append({**identity, "evidence_id": eid, "query": task["question"], "tool_id": tool,
                             "before_rank": pair[0], "after_rank": pair[1],
                             "rescued": pair[0] > cutoff >= pair[1], "intervention": intervention,
                             "probe_source": {"path": "results/whatif.jsonl", "line": line},
                             "provenance": PROVENANCE})
    return evidence


def apply_repairs(rows: list[dict[str, Any]], evidence: list[dict[str, Any]],
                  gold: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply rule 5 only to pending misses after D-010's earlier rules.

    Rows must be the deterministic pre-rewrite classification, not already
    upgraded v1 labels. Evidence key includes query identity and cutoff; each
    intervention remains individually identified within that context.
    """
    by_context: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for e in evidence:
        if e["query_kind"] != QUERY_KIND:
            raise ValueError("Only offline task-question evidence is supported by D-078")
        key = (e["task_id"], e["cap_id"], e["strategy"], e["query_sha256"], e["cutoff"])
        by_context[key].append(e)
    out = []
    for row in rows:
        if row["label"] == DESCRIPTION:
            raise ValueError("Apply D-078 to pre-rewrite rows, not legacy description labels")
        task = gold[row["task_id"]]
        qhash = query_hash(task["question"])
        key = (row["task_id"], row["cap_id"], row["strategy"], qhash, TOP)
        tests = by_context.get(key, [])
        # Recheck the actual crossing rather than trusting a cached boolean flag.
        rescued = [e for e in tests if e["before_rank"] > TOP >= e["after_rank"]]
        status = "rescued" if rescued else "no_rescue" if tests else "not_tested"
        assessment = {"status": status, "strategy": row["strategy"], "query_kind": QUERY_KIND,
                      "query_sha256": qhash, "task_id": task["task_id"], "cutoff": TOP,
                      "agent_search_replayed": False, "evidence": tests}
        result = {**row, "classifier_version": VERSION, "repair_assessment": assessment}
        if row["label"] == RETRIEVAL and row["reason"] == "pending_whatif":
            if rescued:
                result.update(label=DESCRIPTION, reason="same_strategy_offline_repair:" + rescued[0]["evidence_id"])
            else:
                result["reason"] = "whatif_no_rescue_same_strategy" if tests else "whatif_not_tested"
        out.append(result)
    return out

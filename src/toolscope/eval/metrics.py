"""Retrieval metrics. Strict ones via ranx on qrels_strict; lenient, capability-level ones here.

Completeness@k (ToolRet §5.1, from COLT): 1 only if every required capability has an
acceptable tool in the top-k. Capability recall@k: share of capabilities with an acceptable
tool in the top-k (each capability counted once, however many of its tools appear).
"""

import math
from typing import Any

from ranx import Qrels, Run, evaluate
from scipy.stats import ttest_rel

RANX_METRICS = ["recall@5", "recall@10", "ndcg@10", "mrr@10"]


def completeness_at_k(ranked_ids: list[str], capabilities: list[dict[str, Any]], k: int) -> float:
    top = set(ranked_ids[:k])
    return float(all(top & set(c["acceptable"]) for c in capabilities))


def capability_recall_at_k(ranked_ids: list[str], capabilities: list[dict[str, Any]], k: int) -> float:
    top = set(ranked_ids[:k])
    return sum(bool(top & set(c["acceptable"])) for c in capabilities) / len(capabilities)


def bonferroni(p: float, m: int) -> float:
    return min(1.0, p * m)


def paired_p(a: list[float], b: list[float]) -> float:
    """Two-sided paired t-test (as ranx 'student'); 1.0 when the scores never differ."""
    if all(x == y for x, y in zip(a, b)):
        return 1.0
    p = float(ttest_rel(a, b).pvalue)
    return 1.0 if math.isnan(p) else p


def run_from_logs(lines: list[dict[str, Any]], strategy: str) -> dict[str, dict[str, float]]:
    """ranx run from logged top_k. Scores are rank-based so ranx keeps the logged order even on ties."""
    run = {}
    for rec in lines:
        if rec["strategy"] == strategy:
            n = len(rec["top_k"])
            run[rec["task_id"]] = {tid: float(n - i) for i, (tid, _) in enumerate(rec["top_k"])}
    return run


def strict_metrics(qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]],
                   per_query: bool = False) -> dict[str, Any]:
    """Recall@5/10, nDCG@10, MRR@10 on the strict qrels (ranx). Means, or per-query lists in qrels order."""
    q, r = Qrels(qrels), Run(run)
    if not per_query:
        return {m: float(v) for m, v in evaluate(q, r, RANX_METRICS, make_comparable=True).items()}
    out = {}
    for m in RANX_METRICS:
        out[m] = [float(x) for x in evaluate(q, r, m, return_mean=False, make_comparable=True)]
    return out

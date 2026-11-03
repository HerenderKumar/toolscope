"""Turn retrieval log lines into per-strategy results: means, per-query scores, Δ vs baseline,
Bonferroni-corrected paired-test p-values, and Markdown / LaTeX tables."""

import statistics
from typing import Any

from ranx import Qrels, Run, compare

from toolscope.eval.metrics import (RANX_METRICS, bonferroni, capability_recall_at_k, completeness_at_k,
                                    paired_p, run_from_logs, strict_metrics)

CUSTOM_METRICS = ["completeness@5", "completeness@10", "cap_recall@10"]
ALL_METRICS = RANX_METRICS + CUSTOM_METRICS
LABELS = {"recall@5": "R@5", "recall@10": "R@10", "ndcg@10": "nDCG@10", "mrr@10": "MRR@10",
          "completeness@5": "C@5", "completeness@10": "C@10", "cap_recall@10": "CapR@10"}
ALPHA = 0.05


def _custom_per_query(lines: list[dict[str, Any]], strategy: str, gold: dict[str, dict],
                      task_ids: list[str]) -> dict[str, list[float]]:
    ranked = {r["task_id"]: [t for t, _ in r["top_k"]] for r in lines if r["strategy"] == strategy}
    caps = {t: gold[t]["capabilities"] for t in task_ids}
    return {
        "completeness@5": [completeness_at_k(ranked[t], caps[t], 5) for t in task_ids],
        "completeness@10": [completeness_at_k(ranked[t], caps[t], 10) for t in task_ids],
        "cap_recall@10": [capability_recall_at_k(ranked[t], caps[t], 10) for t in task_ids],
    }


def evaluate_logs(lines: list[dict[str, Any]], gold_tasks: list[dict[str, Any]],
                  qrels: dict[str, dict[str, int]], strategies: list[str],
                  baseline: str = "bm25") -> dict[str, dict[str, Any]]:
    gold = {g["task_id"]: g for g in gold_tasks}
    task_ids = sorted(qrels)  # ranx returns per-query scores in sorted query-id order
    runs = {s: run_from_logs(lines, s) for s in strategies}
    res: dict[str, dict[str, Any]] = {}
    for s in strategies:
        per_q = strict_metrics(qrels, runs[s], per_query=True)
        per_q.update(_custom_per_query(lines, s, gold, task_ids))
        own = [r for r in lines if r["strategy"] == s]
        res[s] = {
            "per_query": per_q,
            "mean": {m: statistics.fmean(per_q[m]) for m in ALL_METRICS},
            "tokens_mean": statistics.fmean(r["tokens"] for r in own),
            "latency_ms_median": statistics.median(r["latency_ms"] for r in own),
            "delta": {}, "p_raw": {}, "p_bonf": {},
        }
    others = [s for s in strategies if s != baseline]
    if not others:
        return res
    # strict metrics: ranx paired t-test ("student"); custom metrics: the same test via scipy
    report = compare(Qrels(qrels), [Run(runs[s], name=s) for s in strategies], metrics=RANX_METRICS,
                     stat_test="student", make_comparable=True).to_dict()
    for s in others:
        for m in ALL_METRICS:
            p = (report[s]["comparisons"][baseline][m] if m in RANX_METRICS
                 else paired_p(res[s]["per_query"][m], res[baseline]["per_query"][m]))
            res[s]["delta"][m] = res[s]["mean"][m] - res[baseline]["mean"][m]
            res[s]["p_raw"][m] = float(p)
            res[s]["p_bonf"][m] = bonferroni(float(p), len(others))
    return res


def _cell(res: dict[str, Any], s: str, m: str) -> str:
    mark = "†" if res[s]["p_bonf"].get(m, 1.0) < ALPHA else ""
    return f"{res[s]['mean'][m]:.3f}{mark}"


def render_markdown(res: dict[str, dict[str, Any]], strategies: list[str], n_queries: int,
                    n_comparisons: int, baseline: str = "bm25") -> str:
    head = "| Strategy | " + " | ".join(LABELS[m] for m in ALL_METRICS) + " | Tokens@10 | Latency ms |"
    rows = [head, "|" + "---|" * (len(ALL_METRICS) + 3)]
    for s in strategies:
        cells = " | ".join(_cell(res, s, m) for m in ALL_METRICS)
        rows.append(f"| {s} | {cells} | {res[s]['tokens_mean']:.0f} | {res[s]['latency_ms_median']:.1f} |")
    delta = ["| Strategy | " + " | ".join(f"Δ{LABELS[m]} (p)" for m in ALL_METRICS) + " |",
             "|" + "---|" * (len(ALL_METRICS) + 1)]
    for s in strategies:
        if s == baseline:
            continue
        cells = " | ".join(f"{res[s]['delta'][m]:+.3f} ({res[s]['p_bonf'][m]:.3f})" for m in ALL_METRICS)
        delta.append(f"| {s} | {cells} |")
    return "\n".join([
        f"## Retrieval results ({n_queries} queries)", "",
        f"Strict metrics (R, nDCG, MRR) use `qrels_strict.json`; C@k and CapR@10 use the equivalence "
        f"classes. † = differs from {baseline} at α = {ALPHA} after Bonferroni correction "
        f"({n_comparisons} comparisons, paired t-test). Tokens@10 = mean Qwen2.5 tokens of the top-10 "
        f"tool schemas; latency = median per query, CPU.", "", *rows, "",
        f"## Differences vs {baseline} (Bonferroni-corrected p)", "", *delta, ""])


def render_latex(res: dict[str, dict[str, Any]], strategies: list[str], n_queries: int,
                 baseline: str = "bm25") -> str:
    cols = "l" + "r" * (len(ALL_METRICS) + 1)
    head = " & ".join(["Strategy"] + [LABELS[m] for m in ALL_METRICS] + ["Tok."]) + r" \\"
    body = []
    for s in strategies:
        cells = [_cell(res, s, m).replace("†", r"$^\dagger$") for m in ALL_METRICS]
        body.append(" & ".join([s.replace("_", r"\_")] + cells + [f"{res[s]['tokens_mean']:.0f}"]) + r" \\")
    return "\n".join([
        r"\begin{table}", r"\centering",
        rf"\caption{{Retrieval on LiveMCPBench ({n_queries} tasks, 525 tools). $\dagger$: differs from "
        rf"{baseline} ($p<0.05$, paired t-test, Bonferroni).}}",
        r"\label{tab:retrieval}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule", head, r"\midrule",
        *body, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])

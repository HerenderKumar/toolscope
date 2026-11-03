import pytest

from toolscope.eval.report import evaluate_logs, render_markdown

GOLD = [{"task_id": f"t{i}", "capabilities": [{"acceptable": [f"g{i}"]}]} for i in range(4)]
QRELS = {f"t{i}": {f"g{i}": 1} for i in range(4)}


def line(strategy, i, top):
    return {"strategy": strategy, "task_id": f"t{i}", "top_k": [[t, 1.0] for t in top],
            "tokens": 100, "latency_ms": 5.0}


LINES = ([line("bm25", i, ["x", f"g{i}"]) for i in range(4)] +
         [line("dense", i, [f"g{i}", "x"]) for i in range(4)])


def test_evaluate_logs_means_and_delta_vs_bm25():
    res = evaluate_logs(LINES, GOLD, QRELS, ["bm25", "dense"], baseline="bm25")
    assert res["dense"]["mean"]["mrr@10"] == pytest.approx(1.0)
    assert res["bm25"]["mean"]["mrr@10"] == pytest.approx(0.5)
    assert res["dense"]["delta"]["mrr@10"] == pytest.approx(0.5)
    assert res["dense"]["mean"]["completeness@10"] == 1.0
    assert 0.0 <= res["dense"]["p_bonf"]["mrr@10"] <= 1.0
    assert res["bm25"]["delta"] == {}


def test_render_markdown_has_one_row_per_strategy():
    res = evaluate_logs(LINES, GOLD, QRELS, ["bm25", "dense"], baseline="bm25")
    md = render_markdown(res, ["bm25", "dense"], n_queries=4, n_comparisons=1)
    assert md.count("| bm25 ") >= 1 and md.count("| dense ") >= 2

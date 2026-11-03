import pytest

from toolscope.eval.metrics import (bonferroni, capability_recall_at_k, completeness_at_k, paired_p,
                                    run_from_logs, strict_metrics)

CAPS = [{"acceptable": ["a/x", "b/x"]}, {"acceptable": ["c/y"]}]


def test_completeness_needs_every_capability():
    assert completeness_at_k(["b/x", "z", "c/y"], CAPS, k=3) == 1.0
    assert completeness_at_k(["b/x", "z", "c/y"], CAPS, k=2) == 0.0


def test_capability_recall_counts_each_capability_once():
    assert capability_recall_at_k(["a/x", "b/x", "z"], CAPS, k=3) == 0.5


def test_bonferroni_multiplies_and_caps_at_one():
    assert bonferroni(0.01, 5) == pytest.approx(0.05)
    assert bonferroni(0.4, 5) == 1.0


def test_run_from_logs_uses_logged_order_not_scores():
    lines = [{"strategy": "bm25", "task_id": "t1", "top_k": [["b", 0.0], ["a", 0.0]]}]
    run = run_from_logs(lines, "bm25")
    assert run["t1"]["b"] > run["t1"]["a"]


def test_strict_metrics_on_tiny_example():
    qrels = {"t1": {"a": 1}, "t2": {"b": 1}}
    run = {"t1": {"a": 2.0, "x": 1.0}, "t2": {"x": 2.0, "b": 1.0}}
    m = strict_metrics(qrels, run)
    assert m["mrr@10"] == pytest.approx(0.75)
    assert m["recall@5"] == pytest.approx(1.0)


def test_paired_p_is_one_for_identical_scores():
    assert paired_p([0.5, 1.0, 0.0], [0.5, 1.0, 0.0]) == 1.0

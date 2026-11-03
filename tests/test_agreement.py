import pytest

from toolscope.judge.agreement import agreement_report, cohen_kappa


def test_kappa_known_values():
    # IR book (Manning et al., Table 8.2): 400 items, 300 yes/yes, 70 no/no, 20 + 10 disagreements -> 0.776
    a = ["yes"] * 300 + ["no"] * 70 + ["yes"] * 20 + ["no"] * 10
    b = ["yes"] * 300 + ["no"] * 70 + ["no"] * 20 + ["yes"] * 10
    assert cohen_kappa(a, b) == pytest.approx(0.776, abs=0.001)
    assert cohen_kappa(["s", "f"], ["s", "f"]) == 1.0
    assert cohen_kappa(["s", "s"], ["s", "s"]) == 1.0


def test_agreement_report():
    rep = agreement_report([("success", "success"), ("failure", "success"), ("failure", "failure")])
    assert rep["n"] == 3 and rep["agreement"] == pytest.approx(2 / 3)
    assert rep["confusion"]["human=failure/judge=success"] == 1 and rep["passes"] is False


def test_kappa_interval_brackets_the_estimate():
    pairs = [("success", "success")] * 30 + [("failure", "failure")] * 50 + [("failure", "success")] * 10
    rep = agreement_report(pairs)
    lo, hi = rep["kappa_ci"]
    assert lo <= rep["kappa"] <= hi and hi - lo < 0.5

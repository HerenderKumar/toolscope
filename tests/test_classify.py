from toolscope.diagnosis.classify import (COMPOSITION, ENVIRONMENT, RANKING, RETRIEVAL, SELECTION, broken_tools,
                                          label_capability, label_run, retrieval_label)

CAP_A = {"cap_id": "t#1", "acceptable": ["s/a"]}
CAP_B = {"cap_id": "t#2", "acceptable": ["s/b"]}


def _rec(steps):
    return {"task_id": "t", "steps": steps}


def _call(tool, ok=True, error=None, snippet="x"):
    return {"type": "call", "tool_id": tool, "ok": ok, "error": error, "result_snippet": snippet}


def _search(shortlist):
    return {"type": "search", "query": "q", "shortlist": shortlist}


OFFLINE = {"bm25": {"t": ["x/1"] * 10 + ["s/b"]}, "semantic": {"t": ["s/b", "s/a"]}}


def _label(cap, steps, solved=(), n_caps=1, broken=frozenset(), offline=OFFLINE):
    return label_capability(cap, _rec(steps), "bm25", set(solved), n_caps, set(broken), offline)


def test_solved_and_environment_first():
    assert _label(CAP_A, [_call("s/a")])["label"] == "solved"
    assert _label(CAP_A, [], broken={"s/a"}) == {"label": ENVIRONMENT, "reason": "all_acceptable_broken"}
    out = _label(CAP_A, [_search(["s/a"]), _call("s/a", ok=False, error="ECONNREFUSED")])
    assert out == {"label": ENVIRONMENT, "reason": "gold_call_failed_outside"}


def test_empty_transcript_is_not_solved():
    out = _label(CAP_A, [_call("s/a", snippet='{"charCount": 0}')])
    assert out["label"] == ENVIRONMENT


def test_selection_error_when_shortlisted():
    assert _label(CAP_A, [_search(["s/a"])]) == {"label": SELECTION, "reason": "not_called"}
    out = _label(CAP_A, [_search(["s/a"]), _call("s/a", ok=False, error="invalid arguments: x")])
    assert out == {"label": SELECTION, "reason": "bad_call"}


def test_composition_versus_query():
    offline = {"bm25": {"t": ["s/a"]}, "semantic": {"t": []}}
    assert _label(CAP_A, [], solved={"t#2"}, n_caps=2, offline=offline)["label"] == COMPOSITION
    assert _label(CAP_A, [], n_caps=1, offline=offline) == {"label": SELECTION, "reason": "query"}


def test_retrieval_side_labels():
    assert retrieval_label(CAP_A, "t", "bm25", OFFLINE) == (RANKING, "found_by:semantic")
    assert retrieval_label(CAP_B, "t", "bm25", {"bm25": OFFLINE["bm25"]}) == (RANKING, "rank_11")
    assert retrieval_label(CAP_A, "t", "bm25", {"bm25": OFFLINE["bm25"]}) == (RETRIEVAL, "pending_whatif")
    assert retrieval_label(CAP_B, "t", "semantic", OFFLINE) is None


def test_broken_tools_and_label_run():
    recs = [_rec([_call("s/x", ok=False, error="detected an anomaly"), _call("s/y", ok=False, error="ETIMEDOUT")])]
    assert "s/x" in broken_tools(recs) and "s/y" not in broken_tools(recs)
    gold = {"t": {"capabilities": [CAP_A, CAP_B]}}
    rows = label_run({"t": _rec([_call("s/b")])}, gold, "bm25", set(), {"bm25": {"t": ["s/a"]}})
    assert [r["label"] for r in rows] == [COMPOSITION, "solved"]


def test_known_broken_ok_call_is_not_solved():
    cap = {"cap_id": "t#1", "acceptable": ["kospi-kosdaq/get_stock_market_cap"]}
    out = label_capability(cap, _rec([_call("kospi-kosdaq/get_stock_market_cap", snippet="{}")]), "bm25",
                           set(), 1, broken_tools([]), OFFLINE)
    assert out["label"] == ENVIRONMENT

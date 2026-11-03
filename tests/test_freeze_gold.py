from toolscope.catalog.freeze_gold import equivalence_classes, freeze

CATALOG_IDS = {"filesystem/write_file", "desktop-commander/write_file", "fetch/fetch",
               "biomcp/fetch", "a/x", "b/x", "c/x"}


def draft(*caps):
    return [{"task_id": "t1", "category": "c", "question": "q", "capabilities": list(caps)}]


def cap(n, name, tool_id=None, status="auto"):
    return {"cap_id": f"t1#{n}", "step": n, "annotated_name": name,
            "annotated_tool_id": tool_id, "status": status}


def review(n, resolved):
    return {"cap_id": f"t1#{n}", "resolved_tool_id": resolved}


def eq(ids, equivalent):
    return {"tool_ids": ids, "equivalent": equivalent}


def test_auto_capability_is_strict_and_acceptable():
    gold, qrels, problems = freeze(draft(cap(1, "fetch", "fetch/fetch")), [], [], CATALOG_IDS)
    assert problems == []
    assert gold[0]["capabilities"][0]["strict"] == ["fetch/fetch"]
    assert gold[0]["capabilities"][0]["acceptable"] == ["fetch/fetch"]
    assert qrels == {"t1": {"fetch/fetch": 1}}


def test_equivalent_tools_are_acceptable_but_not_strict():
    eqs = [eq("filesystem/write_file desktop-commander/write_file", "yes")]
    gold, _, problems = freeze(draft(cap(1, "write_file", None, "ambiguous")),
                               [review(1, "filesystem/write_file")], eqs, CATALOG_IDS)
    assert problems == []
    c = gold[0]["capabilities"][0]
    assert c["strict"] == ["filesystem/write_file"]
    assert c["acceptable"] == ["desktop-commander/write_file", "filesystem/write_file"]


def test_non_equivalent_row_adds_nothing():
    gold, _, _ = freeze(draft(cap(1, "fetch", "fetch/fetch")), [],
                        [eq("biomcp/fetch fetch/fetch", "no")], CATALOG_IDS)
    assert gold[0]["capabilities"][0]["acceptable"] == ["fetch/fetch"]


def test_equivalence_is_transitive():
    classes = equivalence_classes([eq("a/x b/x", "yes"), eq("b/x c/x", "yes")])
    assert classes["a/x"] == {"a/x", "b/x", "c/x"}


def test_drop_removes_capability():
    gold, _, problems = freeze(draft(cap(1, "fetch", "fetch/fetch"), cap(2, "nope", None, "unknown")),
                               [review(2, "DROP")], [], CATALOG_IDS)
    assert problems == []
    assert [c["cap_id"] for c in gold[0]["capabilities"]] == ["t1#1"]


def test_unresolved_review_is_a_problem():
    _, _, problems = freeze(draft(cap(1, "nope", None, "unknown")), [review(1, "")], [], CATALOG_IDS)
    assert any("unresolved" in p for p in problems)


def test_unknown_tool_id_is_a_problem():
    _, _, problems = freeze(draft(cap(1, "nope", None, "unknown")), [review(1, "made/up")], [], CATALOG_IDS)
    assert any("unknown_id" in p for p in problems)


def test_blank_equivalence_decision_is_a_problem():
    _, _, problems = freeze(draft(cap(1, "fetch", "fetch/fetch")), [],
                            [eq("biomcp/fetch fetch/fetch", "")], CATALOG_IDS)
    assert any("equivalence_undecided" in p for p in problems)


def test_task_with_every_capability_dropped_is_a_problem():
    _, _, problems = freeze(draft(cap(1, "nope", None, "unknown")), [review(1, "DROP")], [], CATALOG_IDS)
    assert any("empty_task" in p for p in problems)


def test_strict_qrels_load_in_ranx():
    from ranx import Qrels
    _, qrels, _ = freeze(draft(cap(1, "fetch", "fetch/fetch")), [], [], CATALOG_IDS)
    assert Qrels(qrels).to_dict() == {"t1": {"fetch/fetch": 1}}

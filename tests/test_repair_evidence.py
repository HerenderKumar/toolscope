"""D-078 must never transfer rescue evidence to an untested context."""

from copy import deepcopy

import pytest

from toolscope.diagnosis.repair_evidence import apply_repairs, normalize_probes

GOLD = {"t": {"task_id": "t", "question": "find the required item", "capabilities": [
    {"cap_id": "t#1", "acceptable": ["s/a", "s/b"]}]}}
CARDS = {f"s/{i}": {"description": "unused"} for i in range(100)}
CARDS.update({"s/a": {"description": "old a"}, "s/b": {"description": "old b"}})
REWRITES = [{"tool_id": f"s/{t}", "model": "frozen-model", "prompt_version": "rewrite-v1",
             "original": f"old {t}", "rewritten": f"new {t}"} for t in ("a", "b")]
PROBES = [{"task_id": "t", "cap_id": "t#1", "tool_id": "s/a", "bm25": [50, 20],
           "semantic": [70, 3], "moved_into_top10": ["bm25"]}]  # deliberately misleading legacy flag


def row(strategy="bm25", label="retrieval miss", reason="pending_whatif", scope="retrieval"):
    return {"task_id": "t", "cap_id": "t#1", "strategy": strategy, "scope": scope, "agent": "",
            "label": label, "reason": reason}


def test_rescue_does_not_transfer_across_strategies_and_untested_is_not_failed():
    evidence = normalize_probes(PROBES, REWRITES, GOLD, CARDS)
    rows = apply_repairs([row(s) for s in ("bm25", "semantic", "toolret", "hybrid", "rerank", "server_first")],
                         evidence, GOLD)
    assert rows[0]["label"] == "retrieval miss"
    assert rows[0]["repair_assessment"]["status"] == "no_rescue"
    assert rows[1]["label"] == "bad description"
    assert rows[1]["repair_assessment"]["status"] == "rescued"
    for r in rows[2:]:
        assert r["label"] == "retrieval miss" and r["reason"] == "whatif_not_tested"
        assert r["repair_assessment"]["status"] == "not_tested"
        assert not r["repair_assessment"]["evidence"]


def test_query_and_cutoff_must_match_and_an_existing_hit_is_not_a_rescue():
    evidence = normalize_probes(PROBES, REWRITES, GOLD, CARDS)
    for mismatch in ({"query_sha256": "other query"}, {"cutoff": 5}):
        changed = [{**e, **mismatch} for e in evidence]
        assert apply_repairs([row("semantic")], changed, GOLD)[0]["reason"] == "whatif_not_tested"
    already_found = [{**PROBES[0], "semantic": [4, 2]}]
    e = normalize_probes(already_found, REWRITES, GOLD, CARDS)
    assert apply_repairs([row("semantic")], e, GOLD)[0]["label"] == "retrieval miss"


def test_multiple_acceptable_tools_keep_distinct_intervention_evidence():
    probes = PROBES + [{**PROBES[0], "tool_id": "s/b", "bm25": [80, 2], "semantic": [90, 70]}]
    evidence = normalize_probes(probes, REWRITES, GOLD, CARDS)
    result = apply_repairs([row()], evidence, GOLD)[0]
    assert result["label"] == "bad description"
    saved = result["repair_assessment"]["evidence"]
    assert [e["rescued"] for e in saved] == [False, True]
    assert len({e["intervention_id"] for e in saved}) == 2
    assert all(e["intervention"]["prompt_version"] == "rewrite-v1" for e in saved)
    changed = deepcopy(REWRITES)
    changed[0]["rewritten"] = "another frozen edit"
    assert normalize_probes(PROBES, changed, GOLD, CARDS)[0]["evidence_id"] != evidence[0]["evidence_id"]


def test_earlier_precedence_is_preserved_and_agent_evidence_is_explicitly_offline():
    evidence = normalize_probes(PROBES, REWRITES, GOLD, CARDS)
    labels = ("solved", "found", "ranking confusion", "environment failure", "composition error", "agent selection error")
    rows = apply_repairs([row("semantic", l, "earlier_rule", "agent") for l in labels], evidence, GOLD)
    assert [r["label"] for r in rows] == list(labels)
    assert all(r["reason"] == "earlier_rule" for r in rows)
    assert all(r["repair_assessment"]["query_kind"] == "offline_task_question" for r in rows)
    assert all(r["repair_assessment"]["agent_search_replayed"] is False for r in rows)


def test_ambiguous_rewrite_version_and_catalog_mismatch_fail_closed():
    with pytest.raises(ValueError, match="exactly one historical rewrite"):
        normalize_probes(PROBES, REWRITES + [REWRITES[0]], GOLD, CARDS)
    with pytest.raises(ValueError, match="exactly one historical rewrite"):
        normalize_probes(PROBES, [], GOLD, CARDS)
    with pytest.raises(ValueError, match="differs from the catalog"):
        normalize_probes(PROBES, [{**REWRITES[0], "original": "different"}], GOLD, CARDS)


@pytest.mark.parametrize("pair", [[0, 2], [50, -1], [50, 999], [True, 3], [20], "20,3"])
def test_invalid_ranks_cannot_support_a_repair(pair):
    with pytest.raises(ValueError, match="Invalid measured ranks"):
        normalize_probes([{**PROBES[0], "bm25": pair}], REWRITES, GOLD, CARDS)


def test_non_gold_tool_duplicate_probe_and_wrong_query_kind_are_rejected():
    with pytest.raises(ValueError, match="not acceptable"):
        normalize_probes([{**PROBES[0], "tool_id": "s/wrong"}], REWRITES, GOLD, CARDS)
    with pytest.raises(ValueError, match="Duplicate measured intervention"):
        normalize_probes(PROBES * 2, REWRITES, GOLD, CARDS)
    e = normalize_probes(PROBES, REWRITES, GOLD, CARDS)
    with pytest.raises(ValueError, match="offline task-question"):
        apply_repairs([row()], [{**e[0], "query_kind": "agent_query"}], GOLD)
    with pytest.raises(ValueError, match="pre-rewrite rows"):
        apply_repairs([row(label="bad description")], e, GOLD)

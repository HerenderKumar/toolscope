from toolscope.diagnosis.tool_health import description_flags, tool_health


def _card(tid, desc, params=()):
    return {"tool_id": tid, "server_key": tid.split("/")[0], "description": desc, "params": list(params)}


def test_description_flags():
    assert description_flags(_card("s/a", "")) == ["empty_description"]
    assert description_flags(_card("s/a", "获取菜谱")) == ["short_description", "non_english_description"]
    long = "Return the weather forecast for a city over the next days."
    assert description_flags(_card("s/a", long, [{"name": "city", "description": ""}])) == ["undocumented_params"]


def test_tool_health_counts_misses_and_best_whatif():
    cards = [_card("s/a", "x" * 40), _card("s/b", "y" * 40)]
    gold = {"t": {"capabilities": [{"cap_id": "t#1", "acceptable": ["s/a"]}]}}
    labels = [{"scope": "retrieval", "strategy": "bm25", "cap_id": "t#1", "label": "bad description"},
              {"scope": "retrieval", "strategy": "semantic", "cap_id": "t#1", "label": "found"},
              {"scope": "agent", "cap_id": "t#1", "label": "retrieval miss"}]
    whatif = [{"tool_id": "s/a", "bm25": [50, 3], "semantic": [40, 30], "moved_into_top10": ["bm25"]}]
    rows = tool_health(cards, gold, labels, whatif)
    assert rows[0]["tool_id"] == "s/a" and rows[0]["missed"] == 1 and rows[0]["gold_caps"] == 1
    assert rows[0]["whatif"]["gain"] == 47 and rows[1]["whatif"] is None

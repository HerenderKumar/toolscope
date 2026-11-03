from toolscope.servers.drift import compare_tool, tool_drift


def schema(props, required=()):
    return {"type": "object", "properties": {p: {"type": "string"} for p in props}, "required": list(required)}


def test_missing_tool_is_breaking():
    assert compare_tool(schema(["a"], ["a"]), None) == "breaking"


def test_renamed_required_param_is_breaking():
    assert compare_tool(schema(["trainNo"], ["trainNo"]), schema(["trainCode"], ["trainCode"])) == "breaking"


def test_new_required_param_is_breaking():
    assert compare_tool(schema(["a"], ["a"]), schema(["a", "q"], ["a", "q"])) == "breaking"


def test_new_optional_param_is_additive():
    assert compare_tool(schema(["a"], ["a"]), schema(["a", "style"], ["a"])) == "additive"


def test_empty_live_schema_is_unchecked_not_breaking():
    assert compare_tool(schema(["path"], ["path"]), {"$schema": "x"}) == "unchecked"


def test_identical_is_same():
    assert compare_tool(schema(["a"], ["a"]), schema(["a"], ["a"])) == "same"


def test_tool_drift_counts_by_kind():
    catalog = {("s", "t1"): schema(["a"]), ("s", "t2"): schema(["b"], ["b"])}
    live = {("s", "t1"): schema(["a", "x"])}
    out = tool_drift(catalog, live)
    assert out[("s", "t1")] == "additive" and out[("s", "t2")] == "breaking"

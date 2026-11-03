from toolscope.catalog.show_tool import find_tools

CARDS = [
    {"tool_id": "filesystem/write_file", "tool_name": "write_file"},
    {"tool_id": "desktop-commander/write_file", "tool_name": "write_file"},
    {"tool_id": "fetch/fetch", "tool_name": "fetch"},
]


def test_find_by_name_returns_every_server():
    assert [c["tool_id"] for c in find_tools("write_file", CARDS)] == [
        "filesystem/write_file", "desktop-commander/write_file"]


def test_find_by_full_id_returns_one():
    assert [c["tool_id"] for c in find_tools("fetch/fetch", CARDS)] == ["fetch/fetch"]


def test_unknown_name_returns_nothing():
    assert find_tools("nope", CARDS) == []

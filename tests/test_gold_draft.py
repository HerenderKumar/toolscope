from toolscope.catalog.build_gold_draft import build, parse_tool_list, suggest

CARDS = [
    {"tool_id": "filesystem/read_file", "tool_name": "read_file", "server_key": "filesystem"},
    {"tool_id": "desktop-commander/read_file", "tool_name": "read_file", "server_key": "desktop-commander"},
    {"tool_id": "desktop-commander/start_process", "tool_name": "start_process", "server_key": "desktop-commander"},
    {"tool_id": "biomcp/trial_references_getter", "tool_name": "trial_references_getter", "server_key": "biomcp"},
]


def test_parse_tool_list():
    assert parse_tool_list("1. load\n2. beat_track\n\n3) mfcc ") == ["load", "beat_track", "mfcc"]


def test_build_statuses():
    tasks = [{"task_id": "t1", "category": "x", "Question": "q",
              "Annotator Metadata": {"Tools": "1. start_process\n2. read_file\n3. nope", "Number of tools": "3"}}]
    drafts, review, shared = build(tasks, CARDS)
    caps = drafts[0]["capabilities"]
    assert [c["status"] for c in caps] == ["auto", "ambiguous", "unknown"]
    assert caps[0]["annotated_tool_id"] == "desktop-commander/start_process"
    assert caps[1]["cap_id"] == "t1#2"
    assert len(review) == 2
    # the same-task server is suggested first for the ambiguous name
    assert review[0]["suggested"].split()[0] == "desktop-commander/read_file"
    assert shared == {"read_file": ["filesystem/read_file", "desktop-commander/read_file"]}


def test_suggest_unknown_close_name():
    assert suggest("trial_references", [], set(), CARDS)[0] == "biomcp/trial_references_getter"


def test_main_refuses_to_overwrite_hand_edits(tmp_path, monkeypatch):
    import pytest
    from toolscope.catalog import build_gold_draft as mod
    (tmp_path / "review.csv").write_text("cap_id,resolved_tool_id\nt1#1,fetch/fetch\n")
    monkeypatch.setattr(mod, "GOLD_DIR", tmp_path)
    monkeypatch.delenv("TOOLSCOPE_FORCE", raising=False)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        mod.main()
    assert (tmp_path / "review.csv").read_text() == "cap_id,resolved_tool_id\nt1#1,fetch/fetch\n"

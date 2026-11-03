import asyncio

from toolscope.agent.loop import AGENT_TOOLS, run_task


def tc(name, arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


class ScriptedLLM:
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def __call__(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        return self.replies.pop(0)


class FakeGateway:
    def __init__(self):
        self.calls = []

    async def __call__(self, name, args):
        self.calls.append((name, args))
        if name == "search_tools":
            return {"result": [{"tool_id": "fs/list_directory", "description": "list", "input_schema": {}},
                               {"tool_id": "music/load", "description": "load", "input_schema": {}}]}
        return {"tool_id": args["name"], "ok": True, "args_valid": True, "executed": args["execute"],
                "result": "[FILE] a.wav", "error": None}


def run(llm, gw, **kw):
    return asyncio.run(run_task("t1", "What files are in /root/music?", llm, gw, **kw))


def test_agent_exposes_only_query_to_the_model_not_k():
    search = next(t for t in AGENT_TOOLS if t["function"]["name"] == "search_tools")
    assert list(search["function"]["parameters"]["properties"]) == ["query"]


def test_full_episode_records_shortlist_choice_and_answer():
    llm = ScriptedLLM([tc("search_tools", {"query": "list files"}),
                       tc("call_tool", {"name": "fs/list_directory", "args": {"path": "/root/music"}}),
                       {"role": "assistant", "content": "One file: a.wav"}])
    gw = FakeGateway()
    rec = run(llm, gw, k=5, execute=True, max_steps=8)
    assert gw.calls[0] == ("search_tools", {"query": "list files", "k": 5, "task_id": "t1"})
    assert gw.calls[1][1]["execute"] is True and gw.calls[1][1]["task_id"] == "t1"
    call = [s for s in rec["steps"] if s["type"] == "call"][0]
    assert call["tool_id"] == "fs/list_directory" and call["shortlist_before"] == ["fs/list_directory", "music/load"]
    assert call["in_shortlist"] is True and call["args_valid"] is True
    assert rec["chosen_tools"] == ["fs/list_directory"]
    assert rec["final_answer"] == "One file: a.wav" and rec["stop_reason"] == "final"


def test_string_arguments_are_parsed_and_unknown_tool_is_reported_back():
    llm = ScriptedLLM([tc("search_tools", '{"query": "x"}'), tc("delete_everything", {}),
                       {"role": "assistant", "content": "done"}])
    gw = FakeGateway()
    rec = run(llm, gw, k=3, execute=False, max_steps=8)
    assert gw.calls == [("search_tools", {"query": "x", "k": 3, "task_id": "t1"})]
    assert any(s["type"] == "invalid" for s in rec["steps"])
    assert "unknown function" in llm.seen[-1][-1]["content"]


def test_stops_at_max_steps():
    llm = ScriptedLLM([tc("search_tools", {"query": "x"})] * 5)
    rec = run(llm, FakeGateway(), k=5, execute=False, max_steps=3)
    assert rec["stop_reason"] == "max_steps" and rec["n_steps"] == 3


def test_tool_messages_carry_the_tool_call_id():
    llm = ScriptedLLM([{"role": "assistant", "content": "", "tool_calls": [
        {"id": "abc", "function": {"name": "search_tools", "arguments": {"query": "x"}}}]},
        {"role": "assistant", "content": "done"}])
    run(llm, FakeGateway(), k=5, execute=False, max_steps=4)
    tool_msg = [m for m in llm.seen[-1] if m["role"] == "tool"][0]
    assert tool_msg["tool_call_id"] == "abc"
    assert llm.seen[-1][2]["tool_calls"][0]["id"] == "abc"


def test_tool_results_are_cut_at_10000_chars_and_full_length_is_logged():
    # D-063: LiveMCPBench cuts tool results at 10,000 characters (baseline/run_conversation.py:76)
    class LongGateway(FakeGateway):
        async def __call__(self, name, args):
            res = await super().__call__(name, args)
            if name == "call_tool":
                res["result"] = "x" * 25000
            return res
    llm = ScriptedLLM([tc("call_tool", {"name": "fs/read_file", "args": {}}), {"role": "assistant", "content": "ok"}])
    rec = run(llm, LongGateway(), k=5, execute=True, max_steps=4)
    tool_msg = [m for m in llm.seen[-1] if m["role"] == "tool"][0]
    assert len(tool_msg["content"]) == 10000
    call = [s for s in rec["steps"] if s["type"] == "call"][0]
    assert call["result_chars"] > 25000 and call["truncated"] is True


def test_short_results_are_not_marked_truncated():
    llm = ScriptedLLM([tc("search_tools", {"query": "x"}), {"role": "assistant", "content": "ok"}])
    rec = run(llm, FakeGateway(), k=5, execute=False, max_steps=4)
    assert rec["steps"][0]["truncated"] is False and 0 < rec["steps"][0]["result_chars"] < 10000


def test_broken_json_arguments_are_reported_like_livemcpbench_not_called():
    # D-067: a reply cut mid tool call gives unparseable arguments; LiveMCPBench answers "Error: <json error>"
    llm = ScriptedLLM([tc("call_tool", '{"name": "fs/write_file", "args": {"path": "/root/a.md", "content": "lo'),
                       {"role": "assistant", "content": "gave up"}])
    gw = FakeGateway()
    rec = run(llm, gw, k=5, execute=True, max_steps=4)
    assert gw.calls == []
    tool_msg = [m for m in llm.seen[-1] if m["role"] == "tool"][0]
    assert tool_msg["content"].startswith("Error: Unterminated string")
    assert rec["steps"][0]["type"] == "invalid" and "invalid JSON" in rec["steps"][0]["error"]


def test_broken_inner_args_string_is_reported_not_sent_to_the_gateway():
    llm = ScriptedLLM([tc("call_tool", {"name": "fs/write_file", "args": '{"path": "/root/a.md", "content": "lo'}),
                       {"role": "assistant", "content": "gave up"}])
    gw = FakeGateway()
    rec = run(llm, gw, k=5, execute=True, max_steps=4)
    assert gw.calls == []
    call = rec["steps"][0]
    assert call["type"] == "call" and call["args_valid"] is False and call["error"].startswith("Error: Unterminated")

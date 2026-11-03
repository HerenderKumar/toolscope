from toolscope.judge.livemcpeval import judge_messages, parse_verdict, prompts


def test_prompts_are_read_verbatim():
    system_msg, prompt = prompts()
    assert system_msg.startswith("You are an expert in evaluating the performance of a tool-use agent.")
    assert 'Status: "success" or "failure"' in system_msg
    assert prompt.startswith("User Task: \n{task}\n\nKey Points: \n{key_points}")


def test_messages_hold_calls_not_results_or_searches():
    rec = {"final_answer": "done", "steps": [
        {"type": "search", "query": "find weather", "shortlist": ["w/get"]},
        {"type": "call", "tool_id": "w/get", "args": {"city": "Paris"}, "ok": True, "result_snippet": "SECRET-RESULT"}]}
    tools = {"w": {"get": "Get weather."}}
    msgs = judge_messages("Weather in Paris?", "1. get weather", rec, tools)
    text = msgs[1]["content"]
    assert '1. {"server_name": "w", "tool_name": "get", "params": {"city": "Paris"}}' in text
    assert "Server: w\nTool: get\nDescription: Get weather." in text
    assert "SECRET-RESULT" not in text and "find weather" not in text
    assert "Final Response: \ndone" in text


def test_parse_verdict():
    assert parse_verdict('Thoughts: ok\nStatus: "success"') == ("success", "ok")
    assert parse_verdict("Thoughts: no\nStatus: failure")[0] == "failure"
    assert parse_verdict("no status here")[0] == "unparsed"


def test_tool_id_without_server():
    rec = {"final_answer": "", "steps": [{"type": "call", "tool_id": "made_up_tool", "args": {}}]}
    text = judge_messages("q", "k", rec, {})[1]["content"]
    assert '"server_name": "", "tool_name": "made_up_tool"' in text and "Tool made_up_tool not found" in text


def test_status_variant_adds_only_the_status():
    rec = {"final_answer": "x", "steps": [
        {"type": "call", "tool_id": "w/get", "args": {}, "ok": True, "result_snippet": "R1"},
        {"type": "call", "tool_id": "w/get", "args": {}, "ok": False, "error": "tool not found"}]}
    plain = judge_messages("q", "k", rec, {})[1]["content"]
    status = judge_messages("q", "k", rec, {}, with_status=True)[1]["content"]
    assert '"params": {}} [status: OK]' in status and '"params": {}} [status: ERROR]' in status
    assert "tool not found" not in status and "R1" not in status
    assert status.replace(" [status: OK]", "").replace(" [status: ERROR]", "") == plain

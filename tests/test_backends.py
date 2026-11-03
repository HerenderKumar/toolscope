import json

import httpx

from toolscope.agent.backends import from_openai_message, ollama_chat, openai_chat, to_openai_messages

OLLAMA_STYLE = [
    {"role": "system", "content": "sys"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "search_tools", "arguments": {"query": "x"}}}]},
    {"role": "tool", "tool_name": "search_tools", "tool_call_id": "c1", "content": "[]"},
]


def test_to_openai_messages_stringifies_arguments_and_keeps_ids():
    out = to_openai_messages(OLLAMA_STYLE)
    call = out[1]["tool_calls"][0]
    assert call == {"id": "c1", "type": "function", "function": {"name": "search_tools", "arguments": '{"query": "x"}'}}
    assert out[2] == {"role": "tool", "tool_call_id": "c1", "content": "[]"}


def test_from_openai_message_parses_arguments():
    msg = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c9", "type": "function", "function": {"name": "call_tool", "arguments": '{"name": "a/b", "args": {}}'}}]}
    assert from_openai_message(msg) == {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c9", "function": {"name": "call_tool", "arguments": {"name": "a/b", "args": {}}}}]}


def test_openai_chat_posts_to_base_url_with_key_and_counts_usage():
    seen = {}

    def handler(request):
        seen["url"], seen["auth"] = str(request.url), request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "done"}}],
                                         "usage": {"prompt_tokens": 120, "completion_tokens": 7}})
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
    chat = openai_chat("some-model", usage, base_url="https://gw.example/v1", api_key="sk-test",
                       transport=httpx.MockTransport(handler))
    reply = chat([{"role": "user", "content": "hi"}], [])
    assert reply == {"role": "assistant", "content": "done", "tool_calls": []}
    assert seen["url"] == "https://gw.example/v1/chat/completions" and seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "some-model" and seen["body"]["temperature"] == 0.7  # default = LiveMCPBench (D-039)
    assert usage == {"llm_calls": 1, "input_tokens": 120, "output_tokens": 7, "max_prompt_tokens": 120}


def test_openai_chat_sends_the_configured_temperature():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "x"}}], "usage": {}})
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
    openai_chat("m", usage, base_url="https://gw/v1", api_key="k", temperature=0.7,
                transport=httpx.MockTransport(handler))([{"role": "user", "content": "hi"}], [])
    assert seen["body"]["temperature"] == 0.7


def test_ollama_chat_sends_the_configured_temperature():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "x"}})
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
    ollama_chat("m", usage, temperature=0.7, transport=httpx.MockTransport(handler))([{"role": "user", "content": "hi"}], [])
    assert seen["body"]["options"]["temperature"] == 0.7


def test_budget_exceeded_is_detected_from_a_402():
    from toolscope.agent.backends import BudgetExceeded, openai_chat
    import pytest
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
    chat = openai_chat("m", usage, base_url="https://gw/v1", api_key="k", transport=httpx.MockTransport(
        lambda r: httpx.Response(402, json={"error": {"message": "API Key Budget Exceeded"}})))
    with pytest.raises(BudgetExceeded):
        chat([{"role": "user", "content": "hi"}], [])


def test_openai_chat_retries_transient_404_then_succeeds(monkeypatch):
    monkeypatch.setenv("TOOLSCOPE_LLM_RETRY_DELAYS_S", "0,0")
    seen = []
    ok = {"choices": [{"message": {"role": "assistant", "content": "hi"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}

    def handler(req):
        seen.append(1)
        return httpx.Response(404, text="not found") if len(seen) == 1 else httpx.Response(200, json=ok)
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
    chat = openai_chat("m", usage, base_url="https://x/v1", api_key="k", transport=httpx.MockTransport(handler))
    assert chat([{"role": "user", "content": "q"}], [])["content"] == "hi"
    assert len(seen) == 2 and usage["llm_calls"] == 1


def test_openai_chat_gives_up_with_provider_unavailable(monkeypatch):
    import pytest
    from toolscope.agent.backends import ProviderUnavailable
    monkeypatch.setenv("TOOLSCOPE_LLM_RETRY_DELAYS_S", "0,0")
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
    chat = openai_chat("m", usage, base_url="https://x/v1", api_key="k",
                       transport=httpx.MockTransport(lambda req: httpx.Response(503, text="busy")))
    with pytest.raises(ProviderUnavailable):
        chat([{"role": "user", "content": "q"}], [])
    assert usage["llm_calls"] == 0


def test_openai_chat_caps_reply_length_like_livemcpbench():
    # D-063: LiveMCPBench sends max_completion_tokens=4096 (utils/llm_api.py:17)
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "x"}}], "usage": {}})
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
    openai_chat("m", usage, base_url="https://gw/v1", api_key="k",
                transport=httpx.MockTransport(handler))([{"role": "user", "content": "hi"}], [])
    assert seen["body"]["max_completion_tokens"] == 4096


def test_ollama_chat_caps_reply_length_like_livemcpbench():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "x"}})
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
    ollama_chat("m", usage, transport=httpx.MockTransport(handler))([{"role": "user", "content": "hi"}], [])
    assert seen["body"]["options"]["num_predict"] == 4096


def test_client_error_keeps_the_provider_message():
    import pytest

    def handler(request):
        return httpx.Response(400, json={"error": {"message": "prompt was filtered"}})
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
    with pytest.raises(httpx.HTTPStatusError, match="prompt was filtered"):
        openai_chat("m", usage, base_url="https://gw/v1", api_key="k",
                    transport=httpx.MockTransport(handler))([{"role": "user", "content": "hi"}], [])


def test_replies_cut_at_the_token_cap_are_counted():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "length",
                                                      "message": {"role": "assistant", "content": "x"}}], "usage": {}})
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
    openai_chat("m", usage, base_url="https://gw/v1", api_key="k",
                transport=httpx.MockTransport(handler))([{"role": "user", "content": "hi"}], [])
    assert usage["length_stops"] == 1


def test_unparseable_arguments_keep_the_error_and_the_raw_text():
    from toolscope.agent.backends import INVALID_JSON
    msg = from_openai_message({"tool_calls": [{"id": "c1", "function": {"name": "call_tool", "arguments": '{"name": "x'}}]})
    args = msg["tool_calls"][0]["function"]["arguments"]
    assert "Unterminated string" in args[INVALID_JSON]
    back = to_openai_messages([{"role": "assistant", "content": "", "tool_calls": msg["tool_calls"]}])
    assert back[0]["tool_calls"][0]["function"]["arguments"] == '{"name": "x'


def test_ollama_reports_its_context_window_and_largest_prompt():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "x"}, "prompt_eval_count": 900})
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
    ollama_chat("m", usage, num_ctx=32768, transport=httpx.MockTransport(handler))([{"role": "user", "content": "hi"}], [])
    assert seen["body"]["options"]["num_ctx"] == 32768 and usage["max_prompt_tokens"] == 900

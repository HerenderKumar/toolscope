"""LLM backends for the agent loop. Both return Ollama-style assistant messages
({"role", "content", "tool_calls": [{"id", "function": {"name", "arguments": dict}}]}) and add token counts to `usage`.

- ollama_chat: local Ollama (default agent, D-020).
- openai_chat: any OpenAI-compatible endpoint (optional paid comparison, D-021); key from the environment only.
"""

import json
import os
import time
from typing import Any, Callable

import httpx

ChatFn = Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]
SEED = 42
# D-063: as LiveMCPBench (utils/llm_api.py:17, max_completion_tokens=4096)
MAX_COMPLETION_TOKENS = int(os.environ.get("TOOLSCOPE_MAX_COMPLETION_TOKENS", "4096"))
LLM_TIMEOUT_S = float(os.environ.get("TOOLSCOPE_LLM_TIMEOUT_S", "600"))
# Ollama context window; 16,384 before D-068. 10,000-char results + 4,096-token replies need more for long tasks
OLLAMA_NUM_CTX = int(os.environ.get("TOOLSCOPE_OLLAMA_NUM_CTX", "16384"))


class BudgetExceeded(RuntimeError):
    """The paid API refused for lack of budget (HTTP 402): stop the run instead of logging fake task errors."""


class ProviderUnavailable(RuntimeError):
    """The paid API kept failing (404/408/429/5xx or no connection) after all retries: stop without recording the
    task, so resume runs it again (30 Oct: aicredits answered 404 for 29 Qwen3-235B tasks, 20 of them in 30 s)."""


TRANSIENT_STATUS = {404, 408, 409, 425, 429, 500, 502, 503, 504}


def retry_delays() -> list[float]:
    return [float(x) for x in os.environ.get("TOOLSCOPE_LLM_RETRY_DELAYS_S", "5,20,60,120").split(",") if x.strip()]


# D-067: arguments that are not valid JSON (e.g. a reply cut at max_completion_tokens) are kept with the parse error,
# so the loop can answer "Error: <json error>" as LiveMCPBench does (baseline/run_conversation.py:154,175)
INVALID_JSON, RAW_ARGS = "__invalid_json__", "__raw_arguments__"


def _parse(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        out = json.loads(raw or "{}")
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError as e:
        return {INVALID_JSON: str(e), RAW_ARGS: raw}


def _raw_arguments(args: dict[str, Any]) -> str:
    """The arguments as the model sent them: the original text when it was not valid JSON."""
    return args[RAW_ARGS] if RAW_ARGS in args else json.dumps(args)


def to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or None, "tool_calls": [
                {"id": tc.get("id"), "type": "function",
                 "function": {"name": tc["function"]["name"], "arguments": _raw_arguments(tc["function"].get("arguments", {}))}}
                for tc in m["tool_calls"]]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id"), "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def from_openai_message(msg: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": msg.get("content") or "", "tool_calls": [
        {"id": tc.get("id"), "function": {"name": tc["function"]["name"], "arguments": _parse(tc["function"].get("arguments"))}}
        for tc in (msg.get("tool_calls") or [])]}


def ollama_chat(model: str, usage: dict[str, int], base_url: str = "http://localhost:11434",
                num_ctx: int | None = None, temperature: float = 0.7,
                transport: httpx.BaseTransport | None = None) -> ChatFn:
    http = httpx.Client(timeout=LLM_TIMEOUT_S, transport=transport)

    def chat(messages, tools):
        r = http.post(f"{base_url}/api/chat", json={"model": model, "messages": messages, "tools": tools,
                                                    "stream": False, "options": {"temperature": temperature, "seed": SEED,
                                                                                "num_ctx": num_ctx or OLLAMA_NUM_CTX,
                                                                                "num_predict": MAX_COMPLETION_TOKENS}})
        r.raise_for_status()
        body = r.json()
        usage["llm_calls"] += 1
        usage["input_tokens"] += body.get("prompt_eval_count", 0)
        # largest single prompt: near num_ctx means Ollama may have dropped the start of the conversation (D-068)
        usage["max_prompt_tokens"] = max(usage.get("max_prompt_tokens", 0), body.get("prompt_eval_count", 0))
        usage["output_tokens"] += body.get("eval_count", 0)
        return body["message"]
    return chat


def openai_chat(model: str, usage: dict[str, int], *, base_url: str, api_key: str, temperature: float = 0.7,
                transport: httpx.BaseTransport | None = None) -> ChatFn:
    http = httpx.Client(timeout=LLM_TIMEOUT_S, transport=transport, headers={"Authorization": f"Bearer {api_key}"})

    def chat(messages, tools):
        body = {"model": model, "messages": to_openai_messages(messages), "temperature": temperature, "seed": SEED,
                "max_completion_tokens": MAX_COMPLETION_TOKENS}
        if tools:
            body["tools"] = tools
        delays, last = retry_delays(), ""
        for attempt in range(len(delays) + 1):
            try:
                r = http.post(f"{base_url.rstrip('/')}/chat/completions", json=body)
            except httpx.TransportError as e:  # no connection / timeout to the provider
                r, last = None, f"{type(e).__name__}: {e}"
            if r is not None:
                if r.status_code == 402:
                    raise BudgetExceeded(r.text[:300])
                if r.status_code not in TRANSIENT_STATUS:
                    break
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            if attempt < len(delays):
                time.sleep(delays[attempt])
        else:
            raise ProviderUnavailable(last)
        if r.status_code >= 400:  # keep the provider's reason (31 Oct: bare "400 Bad Request" mid-task, cause unknown)
            raise httpx.HTTPStatusError(f"HTTP {r.status_code}: {r.text[:500]}", request=r.request, response=r)
        data = r.json()
        u = data.get("usage") or {}
        usage["llm_calls"] += 1
        usage["input_tokens"] += u.get("prompt_tokens", 0)
        usage["max_prompt_tokens"] = max(usage.get("max_prompt_tokens", 0), u.get("prompt_tokens", 0))
        usage["output_tokens"] += u.get("completion_tokens", 0)
        if data["choices"][0].get("finish_reason") == "length":  # reply cut at max_completion_tokens (D-063)
            usage["length_stops"] = usage.get("length_stops", 0) + 1
        return from_openai_message(data["choices"][0]["message"])
    return chat

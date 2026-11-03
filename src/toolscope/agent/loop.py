"""A small tool-using agent: the LLM sees only the gateway's search_tools and call_tool.

The runner fixes k (the model cannot choose the shortlist size) and the execution mode.
Each step is recorded: shortlists, chosen tool, arguments, argument validity, execution result.
"""

import json
import os
import time
from typing import Any, Awaitable, Callable

from toolscope.agent.backends import INVALID_JSON

SYSTEM_PROMPT = (
    "You complete the user's task by using tools. You have no other abilities. "
    "Step 1: call search_tools with a short description of the capability you need. "
    "Step 2: call call_tool with the tool_id of the best result and arguments matching its input_schema. "
    "Repeat for every step of the task. Only when all steps are done, reply with the final answer.")

AGENT_TOOLS = [
    {"type": "function", "function": {
        "name": "search_tools",
        "description": "Search the catalog of available tools. Returns the best tools with tool_id, "
                       "description and input_schema.",
        "parameters": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string", "description": "Short description of the capability you need"}}}}},
    {"type": "function", "function": {
        "name": "call_tool",
        "description": "Run a tool found by search_tools.",
        "parameters": {"type": "object", "required": ["name", "args"], "properties": {
            "name": {"type": "string", "description": "tool_id from search_tools"},
            "args": {"type": "object", "description": "Arguments matching the tool's input_schema"}}}}},
]

ChatFn = Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]
GatewayFn = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
# D-063: LiveMCPBench cuts tool results at 10,000 characters (baseline/run_conversation.py:76); 4,000 before 31 Oct
MAX_TOOL_RESULT_CHARS = int(os.environ.get("TOOLSCOPE_MAX_TOOL_RESULT_CHARS", "10000"))


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError) as e:
        return {INVALID_JSON: str(e)}


async def run_task(task_id: str, question: str, chat: ChatFn, gateway: GatewayFn, *, k: int,
                   execute: bool, max_steps: int) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT},
                                      {"role": "user", "content": question}]
    steps: list[dict[str, Any]] = []
    shortlist: list[str] = []
    final_answer, stop_reason = None, "max_steps"
    start = time.perf_counter()
    for _ in range(max_steps):
        reply = chat(messages, AGENT_TOOLS)
        calls = reply.get("tool_calls") or []
        messages.append({k_: v for k_, v in reply.items() if k_ in ("role", "content", "tool_calls")})
        if not calls:
            final_answer, stop_reason = (reply.get("content") or "").strip(), "final"
            break
        for i, tc in enumerate(calls):
            tc.setdefault("id", f"call_{len(steps)}_{i}")  # OpenAI-style APIs need ids to pair results
            fn = tc.get("function", {})
            name, args = fn.get("name"), _parse_args(fn.get("arguments"))
            if INVALID_JSON in args:  # D-067: as LiveMCPBench, report the JSON error instead of calling anything
                steps.append({"type": "invalid", "name": name, "arguments": None,
                              "error": f"invalid JSON arguments: {args[INVALID_JSON]}"})
                content = f"Error: {args[INVALID_JSON]}"
            elif name == "search_tools":
                res = await gateway("search_tools", {"query": str(args.get("query", "")), "k": k, "task_id": task_id})
                shortlist = [t["tool_id"] for t in res["result"]]
                steps.append({"type": "search", "query": args.get("query"), "shortlist": shortlist})
                content = json.dumps(res["result"], ensure_ascii=False)
            elif name == "call_tool":
                tool_id, tool_args = str(args.get("name", "")), _parse_args(args.get("args"))
                if INVALID_JSON in tool_args:
                    res = {"args_valid": False, "executed": False, "ok": False, "result": None,
                           "error": f"Error: {tool_args[INVALID_JSON]}"}
                else:
                    res = await gateway("call_tool", {"name": tool_id, "args": tool_args, "task_id": task_id,
                                                      "execute": execute})
                steps.append({"type": "call", "tool_id": tool_id, "args": tool_args,
                              "shortlist_before": list(shortlist), "in_shortlist": tool_id in shortlist,
                              "args_valid": res.get("args_valid"), "executed": res.get("executed"),
                              "ok": res.get("ok"), "error": res.get("error"),
                              "result_snippet": (res.get("result") or "")[:500]})
                content = json.dumps({k_: res.get(k_) for k_ in ("ok", "result", "error")}, ensure_ascii=False)
            else:
                steps.append({"type": "invalid", "name": name, "arguments": args})
                content = f"Error: unknown function {name!r}. Use search_tools or call_tool."
            steps[-1].update(result_chars=len(content), truncated=len(content) > MAX_TOOL_RESULT_CHARS)
            messages.append({"role": "tool", "tool_name": name, "tool_call_id": tc["id"],
                             "content": content[:MAX_TOOL_RESULT_CHARS]})
    n_steps = sum(1 for m in messages if m["role"] == "assistant")
    return {"task_id": task_id, "steps": steps, "chosen_tools": [s["tool_id"] for s in steps if s["type"] == "call"],
            "final_answer": final_answer, "stop_reason": stop_reason, "n_steps": n_steps,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1)}

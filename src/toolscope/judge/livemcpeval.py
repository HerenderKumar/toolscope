"""LiveMCPEval task-success judge (D-071/D-072): LiveMCPBench's own prompt, read verbatim from
third_party/LiveMCPBench/evaluator/llm_as_judge_baseline.py (function livemcp_eval) so it cannot drift.

Inputs mirror theirs: the task, the annotators' Steps as key points (their default, not --auto_key_points), the
final response, the tool calls as `execute-tool` argument JSON ({server_name, tool_name, params}) and the
descriptions of the called tools. Like theirs, the judge sees no tool results and no search_tools calls.
"""

import ast
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from toolscope.retrievers.base import LIVEMCPBENCH_DIR

JUDGE_SOURCE = LIVEMCPBENCH_DIR / "evaluator" / "llm_as_judge_baseline.py"
TOOLS_JSON = LIVEMCPBENCH_DIR / "tools" / "LiveMCPTool" / "tools.json"
STATUS = re.compile(r"Status:\s*([\S]+)", re.DOTALL)  # their judge_pattern


@lru_cache(maxsize=None)
def prompts(path: Path = JUDGE_SOURCE) -> tuple[str, str]:
    """(system_msg, prompt template) exactly as assigned inside livemcp_eval()."""
    tree = ast.parse(Path(path).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "livemcp_eval")
    found = {n.targets[0].id: n.value.value for n in fn.body
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id in ("system_msg", "prompt") and isinstance(n.value, ast.Constant)}
    return found["system_msg"], found["prompt"]


@lru_cache(maxsize=None)
def tool_map(path: Path = TOOLS_JSON) -> dict[str, dict[str, str]]:
    """{server_name: {tool_name: description}} built the way their evaluator builds it."""
    out: dict[str, dict[str, str]] = {}
    for server in json.loads(Path(path).read_text()):
        for tool in server["tools"].values():
            for tl in tool["tools"]:
                out.setdefault(tool["server_name"], {})[tl["name"]] = tl["description"]
    return out


def format_tool_description(tools: dict[str, dict[str, str]], server: str, name: str) -> str:
    """Their format_tool_descriptions() for one call."""
    if server not in tools or name not in tools[server]:
        return f"Tool {name} not found in server {server}."
    return f"Server: {server}\nTool: {name}\nDescription: {tools[server][name]}\n\n".strip()


def judge_messages(question: str, key_points: str, rec: dict[str, Any],
                   tools: dict[str, dict[str, str]], with_status: bool = False) -> list[dict[str, str]]:
    """Messages for one trajectory. Content is sent as plain strings (same text as their content-part list).
    with_status (D-074 variant): each tool-call line ends with " [status: OK]" or " [status: ERROR]"; nothing else
    changes."""
    system_msg, prompt = prompts()
    calls, descriptions = [], ""
    for step in rec["steps"]:
        if step.get("type") != "call":
            continue
        server, _, name = step["tool_id"].rpartition("/")  # an invented name may have no server part
        call = json.dumps({"server_name": server, "tool_name": name, "params": step.get("args") or {}},
                          ensure_ascii=False)
        calls.append(call + (f" [status: {'OK' if step.get('ok') else 'ERROR'}]" if with_status else ""))
        descriptions += format_tool_description(tools, server, name)  # appended per call, as theirs
    text = prompt.format(task=question, tool_calls="\n".join(f"{i + 1}. {c}" for i, c in enumerate(calls)),
                         response=rec.get("final_answer") or "", key_points=key_points,
                         tool_descriptions=descriptions)
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": text}]


def parse_verdict(text: str) -> tuple[str, str]:
    """('success' | 'failure' | 'unparsed', thoughts), following their parsing."""
    m = STATUS.search(text)
    judge = m.group(1).strip() if m else text
    thoughts = text.split("Status:")[0].replace("Thoughts:", "").strip()
    if "success" in judge.lower():
        return "success", thoughts
    if "failure" in judge.lower():
        return "failure", thoughts
    return "unparsed", thoughts

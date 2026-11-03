"""End-to-end smoke test: start the real gateway over stdio, search, call a real container tool.

Run:  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/smoke_gateway.py
Fallback check:  TOOLSCOPE_SEMANTIC_MODEL=/nonexistent ... scripts/smoke_gateway.py   (expect fallback=True)
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

LOG_DIR = Path(os.environ.get("TOOLSCOPE_LOG_DIR", "logs"))
QUERY = "What files are in my /root/music folder?"


def count(name: str) -> int:
    p = LOG_DIR / name
    return len(p.read_text().splitlines()) if p.exists() else 0


async def main() -> None:
    before = (count("gateway.jsonl"), count("gateway_calls.jsonl"))
    transport = StdioTransport(command=sys.executable, args=["-m", "toolscope.gateway.server"],
                               env={**os.environ}, cwd=os.getcwd())
    async with Client(transport) as c:
        tools = [t.name for t in await c.list_tools()]
        found = (await c.call_tool("search_tools", {"query": QUERY, "k": 5})).structured_content["result"]
        called = (await c.call_tool("call_tool", {"name": "filesystem/list_directory",
                                                  "args": {"path": "/root/music"}})).structured_content
    last = json.loads((LOG_DIR / "gateway.jsonl").read_text().splitlines()[-1])
    print("gateway tools:", tools)
    print("search top-5:", [t["tool_id"] for t in found])
    print("call_tool:", {k: called[k] for k in ("ok", "args_valid")}, "->", (called["result"] or called["error"])[:80])
    print(f"search_lines_added={count('gateway.jsonl') - before[0]} call_lines_added={count('gateway_calls.jsonl') - before[1]} "
          f"strategy={last['strategy']} fallback={last['fallback_used']} results={len(found)}")


if __name__ == "__main__":
    asyncio.run(main())

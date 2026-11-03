"""Look up what a tool does: server, full description, parameters and the server's docs link.

Run:  PYTHONPATH=src .venv/bin/python -m toolscope.catalog.show_tool write_file
      PYTHONPATH=src .venv/bin/python -m toolscope.catalog.show_tool fetch/fetch
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

LIVEMCPBENCH_DIR = Path(os.environ.get("TOOLSCOPE_LIVEMCPBENCH_DIR", "third_party/LiveMCPBench"))
CATALOG = Path(os.environ.get("TOOLSCOPE_CATALOG", "data/catalog/tools.jsonl"))


def find_tools(query: str, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A full tool_id matches one tool; a bare name matches it on every server."""
    key = "tool_id" if "/" in query else "tool_name"
    return [c for c in cards if c[key] == query]


def server_links() -> dict[str, str]:
    servers = json.loads((LIVEMCPBENCH_DIR / "tools/LiveMCPTool/tools.json").read_text())
    return {next(iter(s["tools"])): s.get("web") or "" for s in servers}


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: show_tool <tool_name | server_key/tool_name>")
    cards = [json.loads(line) for line in CATALOG.read_text().splitlines()]
    matches = find_tools(sys.argv[1], cards)
    links = server_links()
    for c in matches:
        print(f"=== {c['tool_id']}   (server: {c['server_name']}, category: {c['category']})")
        print(f"docs: {links.get(c['server_key'], '')}")
        print(c["description"] or "(no description)")
        print("parameters (* = required):")
        for p in c["params"]:
            star = "*" if p["required"] else " "
            print(f"  {star} {p['name']} ({p['type']}): {p['description'] or '(no description)'}")
        print()
    print(f"matches={len(matches)}")


if __name__ == "__main__":
    main()

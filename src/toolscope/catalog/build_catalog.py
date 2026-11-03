"""Build data/catalog/tools.jsonl: one standard card per LiveMCPBench tool.

Run:  .venv/bin/python -m toolscope.catalog.build_catalog   (with PYTHONPATH=src)
"""

import json
import os
from pathlib import Path
from typing import Any

from toolscope.log import get_logger

log = get_logger("toolscope.catalog")

LIVEMCPBENCH_DIR = Path(os.environ.get("TOOLSCOPE_LIVEMCPBENCH_DIR", "third_party/LiveMCPBench"))
TOOLS_JSON = LIVEMCPBENCH_DIR / "tools" / "LiveMCPTool" / "tools.json"
SERVER_HEALTH = Path(os.environ.get("TOOLSCOPE_SERVER_HEALTH", "data/raw/server_health.json"))
OUT_PATH = Path(os.environ.get("TOOLSCOPE_CATALOG", "data/catalog/tools.jsonl"))


def extract_params(input_schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Name, type, description and required flag for each top-level parameter."""
    required = set(input_schema.get("required") or [])
    params = []
    for name, spec in (input_schema.get("properties") or {}).items():
        spec = spec if isinstance(spec, dict) else {}
        params.append({
            "name": name,
            "type": spec.get("type"),
            "description": (spec.get("description") or "").strip(),
            "required": name in required,
        })
    return params


def make_search_text(server_name: str, tool_name: str, description: str,
                     params: list[dict[str, Any]]) -> str:
    """Server name + tool name + description + parameter names/descriptions."""
    parts = [server_name, tool_name.replace("_", " "), tool_name, description]
    for p in params:
        parts.append(f"{p['name']}: {p['description']}" if p["description"] else p["name"])
    return " | ".join(x for x in parts if x)


def load_server_health(path: Path) -> dict[str, bool]:
    """{server_key: passed_tool_check}. Empty if the tool check hasn't run yet."""
    if not path.exists():
        log.warning("server health file missing; server_ok will be null",
                    extra={"ctx": {"path": str(path)}})
        return {}
    return json.loads(path.read_text())["servers"]


def build_cards(servers: list[dict[str, Any]], health: dict[str, bool]) -> list[dict[str, Any]]:
    cards = []
    for server in servers:
        for server_key, block in server["tools"].items():
            for tool in block["tools"]:
                description = (tool.get("description") or "").strip()
                input_schema = tool.get("inputSchema") or {}
                params = extract_params(input_schema)
                cards.append({
                    "tool_id": f"{server_key}/{tool['name']}",
                    "server_key": server_key,
                    "server_name": server["name"],
                    "server_description": (server.get("description") or "").strip(),
                    "category": server.get("category"),
                    "tool_name": tool["name"],
                    "description": description,
                    "params": params,
                    "search_text": make_search_text(server["name"], tool["name"], description, params),
                    "input_schema": input_schema,
                    "server_ok": health.get(server_key),
                })
    return cards


def main() -> None:
    servers = json.loads(TOOLS_JSON.read_text())
    cards = build_cards(servers, load_server_health(SERVER_HEALTH))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for card in cards:
            f.write(json.dumps(card, ensure_ascii=False) + "\n")
    unique_ids = len({c["tool_id"] for c in cards})
    empty = sum(not c["description"] for c in cards)
    ok = sum(c["server_ok"] is True for c in cards)
    print(f"servers={len(servers)} tools={len(cards)} unique_ids={unique_ids} "
          f"empty_descriptions={empty} tools_on_ok_servers={ok} out={OUT_PATH}")


if __name__ == "__main__":
    main()

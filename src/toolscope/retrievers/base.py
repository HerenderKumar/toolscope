"""Shared retriever interface: retrieve(query, k) -> [(tool_id, score)], best first (D-016)."""

import json
import os
from pathlib import Path
from typing import Any, Protocol

CATALOG = Path(os.environ.get("TOOLSCOPE_CATALOG", "data/catalog/tools.jsonl"))
INDEX_ROOT = Path(os.environ.get("TOOLSCOPE_INDEX_DIR", "indexes"))
LIVEMCPBENCH_DIR = Path(os.environ.get("TOOLSCOPE_LIVEMCPBENCH_DIR", "third_party/LiveMCPBench"))

STRATEGIES = ["bm25", "semantic", "toolret", "hybrid", "rerank", "server_first"]

Ranking = list[tuple[str, float]]


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int) -> Ranking: ...


def load_cards(path: Path = CATALOG) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def dense_doc_text(card: dict[str, Any]) -> str:
    """ToolRet training format: a Python-dict string of name, description, parameters."""
    params = {p["name"]: {"description": p["description"], "type": p["type"]} for p in card["params"]}
    return str({"name": card["tool_name"], "description": card["description"], "parameters": params})


def server_descriptions() -> dict[str, str]:
    """{server_key: server description} from the LiveMCPBench toolset."""
    servers = json.loads((LIVEMCPBENCH_DIR / "tools/LiveMCPTool/tools.json").read_text())
    return {next(iter(s["tools"])): (s.get("description") or s["name"]).strip() for s in servers}

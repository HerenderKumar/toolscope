"""ToolScope gateway: a stateless FastMCP server exposing search_tools and call_tool.

Instrumentation, not a contribution: every search writes one line in the fixed 10-field schema
to logs/gateway.jsonl; every call writes one line to logs/gateway_calls.jsonl.
If the configured retriever fails, search falls back to BM25 and logs fallback_used=true.

Run (stdio, e.g. for the MCP Inspector):
  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python -m toolscope.gateway.server
Over HTTP (Docker Compose): TOOLSCOPE_GATEWAY_TRANSPORT=http TOOLSCOPE_GATEWAY_PORT=8001 ...
"""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fastmcp import FastMCP
from jsonschema import Draft7Validator

from toolscope.log import get_logger
from toolscope.retrievers.runlog import make_record, schema_tokens

log = get_logger("toolscope.gateway")

STRATEGY = os.environ.get("TOOLSCOPE_STRATEGY", "semantic")
CONTAINER = os.environ.get("TOOLSCOPE_CONTAINER", "LiveMCPBench_container")
SERVER_CONFIG = Path(os.environ.get("TOOLSCOPE_SERVER_CONFIG", "data/raw/server_config.json"))
LOG_DIR = Path(os.environ.get("TOOLSCOPE_LOG_DIR", "logs"))
QRELS = Path(os.environ.get("TOOLSCOPE_QRELS", "data/gold/qrels_strict.json"))
CALL_TIMEOUT_S = float(os.environ.get("TOOLSCOPE_CALL_TIMEOUT_S", "300"))  # D-029
ARXIV_MIN_INTERVAL_S = float(os.environ.get("TOOLSCOPE_ARXIV_MIN_INTERVAL_S", "3"))
RETRY_DELAYS_S = tuple(float(x) for x in os.environ.get("TOOLSCOPE_RETRY_DELAYS_S", "5,20").split(",") if x)  # D-042
ENV_FILE = Path(os.environ.get("TOOLSCOPE_ENV_FILE", ".env"))
FALLBACK = "bm25"
# hosts measured unreachable from the evaluation network (D-041): fail fast instead of a 300 s timeout
UNREACHABLE_HOSTS = tuple(h for h in os.environ.get("TOOLSCOPE_UNREACHABLE_HOSTS",
                                                     "baidu.com,weibo.com,163.com").split(",") if h)
ADDRESS_KEYS = ("url", "host", "domain", "site", "link", "uri")  # only arguments that name a network address


def blocked_host(args: Any) -> str | None:
    """An unreachable host named in an address argument (url, domain, …), never in file contents or paths (D-060)."""
    def walk(value: Any, key: str = "") -> str | None:
        if isinstance(value, dict):
            return next((h for k, v in value.items() if (h := walk(v, str(k)))), None)
        if isinstance(value, list):
            return next((h for v in value if (h := walk(v, key))), None)
        if isinstance(value, str) and (any(a in key.lower() for a in ADDRESS_KEYS) or
                                       value.lstrip().lower().startswith(("http://", "https://"))):
            return next((h for h in UNREACHABLE_HOSTS if h in value), None)
        return None
    return walk(args)


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """JSON-Schema errors for `args` against a tool's inputSchema (empty list = valid)."""
    return [e.message for e in Draft7Validator(schema or {"type": "object"}).iter_errors(args)]


def _append(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def create_gateway(*, get_retriever: Callable[[str], Any], pool: Any, cards: dict[str, dict[str, Any]],
                   strategy: str, qrels: dict[str, dict[str, int]], log_dir: Path,
                   count_tokens: Callable[[str], int], run_id: str | None = None) -> FastMCP:
    run_id = run_id or datetime.now(timezone.utc).strftime("gateway-%Y%m%dT%H%M%SZ")
    n_tools = len(cards)

    @asynccontextmanager
    async def lifespan(_server):
        try:
            yield {}
        finally:
            await pool.aclose()

    mcp = FastMCP("toolscope-gateway", lifespan=lifespan, instructions=(
        "Find tools with search_tools(query, k), then run one with call_tool(name, args), "
        "where name is the tool_id from the search results."))

    def rank(query: str) -> tuple[str, list[tuple[str, float]], bool]:
        try:
            return strategy, get_retriever(strategy).retrieve(query, n_tools), False
        except Exception:
            log.exception("retriever failed; falling back to BM25", extra={"ctx": {"strategy": strategy}})
            return FALLBACK, get_retriever(FALLBACK).retrieve(query, n_tools), True

    @mcp.tool
    def search_tools(query: str, k: int = 5, task_id: str | None = None) -> list[dict[str, Any]]:
        """Search the tool catalog. Returns the k best tools: tool_id, description and input_schema."""
        start = time.perf_counter()
        used, full, fallback = rank(query)
        latency = (time.perf_counter() - start) * 1000
        top = full[:k]
        # tool_id is the only identifier shown: a separate bare "name" field made agents call
        # call_tool("load") instead of call_tool("music-analysis/load") (D-026)
        results = [{"tool_id": t, "server": cards[t]["server_key"],
                    "description": cards[t]["description"], "input_schema": cards[t]["input_schema"],
                    "score": round(s, 6)} for t, s in top]
        rec = make_record(run_id, used, task_id, k, full, (qrels.get(task_id) or {}) if task_id else {},
                          latency, schema_tokens(cards, [t for t, _ in top], count_tokens),
                          fallback_used=fallback, query_id=f"{run_id}/{uuid.uuid4().hex[:12]}")
        _append(log_dir / "gateway.jsonl", rec)
        return results

    @mcp.tool
    async def call_tool(name: str, args: dict[str, Any] | None = None, task_id: str | None = None,
                        execute: bool = True) -> dict[str, Any]:
        """Run a tool found by search_tools. `name` is its tool_id (server/tool); `args` must match its input_schema.
        execute=false: selection-only mode, arguments are validated and logged but the tool is not run."""
        args = args or {}
        start = time.perf_counter()
        out: dict[str, Any] = {"tool_id": name, "ok": False, "args_valid": None, "executed": False,
                               "result": None, "error": None}
        card = cards.get(name)
        if card is None:
            out["error"] = f"unknown tool {name!r}; use a tool_id returned by search_tools"
        else:
            errors = validate_args(card["input_schema"], args)
            out["args_valid"] = not errors
            blocked = blocked_host(args)
            if blocked and not errors:
                out["error"] = f"Host unreachable from the evaluation network: {blocked} (D-041)"
            elif errors:
                out["error"] = "invalid arguments: " + "; ".join(errors)
            elif not execute:
                out["ok"], out["result"] = True, "(selection-only mode: arguments valid, tool not executed)"
            else:
                try:
                    out["executed"] = True
                    out["result"] = await pool.call(card["server_key"], card["tool_name"], args)
                    out["ok"] = True
                except Exception as e:
                    out["error"] = f"{type(e).__name__}: {e}"[:2000]
        _append(log_dir / "gateway_calls.jsonl", {
            "call_id": f"{run_id}/{uuid.uuid4().hex[:12]}", "run_id": run_id, "task_id": task_id,
            "tool_id": name, "args": args, "args_valid": out["args_valid"], "executed": out["executed"], "ok": out["ok"],
            "error": out["error"], "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        return out

    return mcp


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines (e.g. COINCAP_API_KEY) into os.environ without overriding existing values."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def build_default() -> FastMCP:
    from transformers import AutoTokenizer

    from toolscope.gateway.downstream import DownstreamPool
    from toolscope.retrievers.base import load_cards
    from toolscope.retrievers.registry import get_retriever

    load_env_file(ENV_FILE)
    cards = {c["tool_id"]: c for c in load_cards()}
    specs = {k: v for s in json.loads(SERVER_CONFIG.read_text()) for k, v in s["config"]["mcpServers"].items()}
    tok = AutoTokenizer.from_pretrained(os.environ.get("TOOLSCOPE_TOKENIZER", "Qwen/Qwen2.5-3B-Instruct"))
    qrels = json.loads(QRELS.read_text()) if QRELS.exists() else {}
    # arXiv API terms: at most 1 request every 3 seconds, per client (D-036)
    arxiv = {"arxiv-mcp-server": "arxiv", "mcp-simple-arxiv": "arxiv"}
    pool = DownstreamPool(CONTAINER, specs, CALL_TIMEOUT_S, min_interval_s={k: ARXIV_MIN_INTERVAL_S for k in arxiv},
                          rate_group=arxiv, retry_delays_s=RETRY_DELAYS_S)
    return create_gateway(get_retriever=get_retriever, pool=pool,
                          cards=cards, strategy=STRATEGY, qrels=qrels, log_dir=LOG_DIR,
                          count_tokens=lambda s: len(tok.encode(s)))


if __name__ == "__main__":
    # stdio by default (MCP Inspector, run_agent.py); TOOLSCOPE_GATEWAY_TRANSPORT=http serves it over HTTP (Docker Compose)
    transport = os.environ.get("TOOLSCOPE_GATEWAY_TRANSPORT", "stdio")
    if transport == "stdio":
        build_default().run(show_banner=False)
    else:
        build_default().run(transport=transport, show_banner=False,
                            host=os.environ.get("TOOLSCOPE_GATEWAY_HOST", "0.0.0.0"),
                            port=int(os.environ.get("TOOLSCOPE_GATEWAY_PORT", "8001")))

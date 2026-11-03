"""Async connections to the LiveMCPBench servers, each run inside the container via `docker exec -i` (D-019)."""

import asyncio
import json
import os
import re
import time
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from toolscope.log import get_logger

log = get_logger("toolscope.gateway.downstream")

PLACEHOLDER = re.compile(r"^\$\{(\w+)\}$")
BOOT = "source /root/.nvm/nvm.sh >/dev/null 2>&1; export PATH=/root/.local/bin:$PATH; cd /root; "


def uv_offline(spec: dict[str, Any]) -> bool:
    """D-071 (proposed, off by default): start `uv`/`uvx` servers from the snapshot's package cache instead of
    re-resolving on PyPI at every start (PyPI stalls → "Connection closed"). Servers that cannot run offline stay
    online: `-n`/`--no-cache` (music-analysis), `git+` sources (weather), and TOOLSCOPE_UV_ONLINE_PACKAGES (biomcp)."""
    if os.environ.get("TOOLSCOPE_UV_OFFLINE", "0") != "1" or spec.get("command") not in ("uv", "uvx"):
        return False
    online = set(os.environ.get("TOOLSCOPE_UV_ONLINE_PACKAGES", "biomcp-python").split(","))
    args = [str(a) for a in spec.get("args", [])]
    return not any(a in ("-n", "--no-cache") or a.startswith("git+") or a in online for a in args)


def docker_args(container: str, spec: dict[str, Any]) -> list[str]:
    """`docker exec` arguments that start one MCP server over stdio inside the container.
    A `${VAR}` env value becomes `-e VAR`, so docker copies it from our environment (no secret in argv)."""
    env_flags: list[str] = ["-e", "UV_OFFLINE=1"] if uv_offline(spec) else []
    for key, value in (spec.get("env") or {}).items():
        m = PLACEHOLDER.match(str(value))
        env_flags += ["-e", m.group(1) if m else f"{key}={value}"]
    command = " ".join(["exec", spec["command"], *[json.dumps(str(a)) for a in spec.get("args", [])]])
    return ["exec", "-i", *env_flags, container, "bash", "-c", BOOT + command]


def docker_env(spec: dict[str, Any], environ: dict[str, str]) -> dict[str, str]:
    """Values for the `-e VAR` secrets of `docker_args`. The MCP stdio client starts `docker` with a
    filtered environment, so these must be handed over explicitly or docker copies nothing (D-031)."""
    out = {}
    for value in (spec.get("env") or {}).values():
        m = PLACEHOLDER.match(str(value))
        if m and m.group(1) in environ:
            out[m.group(1)] = environ[m.group(1)]
    return out


# temporary upstream overloads worth retrying: an error ending in HTTP 429/5xx (e.g. Overpass "...: 502"),
# or a result that reports "HTTP 429/503" from its upstream API (arXiv). Not timeouts: those already took 300 s.
TRANSIENT_ERROR = re.compile(r"[:\s](429|50[0234])\s*$|too many requests|\bHTTP (429|50[0234])\b", re.I)
TRANSIENT_IN_RESULT = re.compile(r"\bHTTP (429|50[0234])\b", re.I)


class DownstreamError(RuntimeError):
    pass


class DownstreamPool:
    """One lazily opened session per server, kept for the gateway's lifetime."""

    def __init__(self, container: str, specs: dict[str, dict[str, Any]], timeout_s: float = 120.0,
                 min_interval_s: dict[str, float] | None = None, rate_group: dict[str, str] | None = None,
                 retry_delays_s: tuple[float, ...] = ()):
        self.container, self.specs, self.timeout_s = container, specs, timeout_s
        self.retry_delays_s = retry_delays_s  # back-off for temporary overloads (D-042)
        self._clients: dict[str, Client] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # polite pacing for APIs with a published rate limit (arXiv: 1 request / 3 s; D-036);
        # servers in the same group share one clock because the limit is per client IP
        self.min_interval_s = min_interval_s or {}
        self.rate_group = rate_group or {}
        self._last_call: dict[str, float] = {}
        self._clock = time.monotonic

    async def _sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    async def _pace(self, server_key: str) -> None:
        interval = self.min_interval_s.get(server_key)
        if not interval:
            return
        group = self.rate_group.get(server_key, server_key)
        wait = self._last_call.get(group, -1e9) + interval - self._clock()
        if wait > 0:
            await self._sleep(wait)
        self._last_call[group] = self._clock()

    async def _new_client(self, spec: dict[str, Any]) -> Client:
        client = Client(StdioTransport(command="docker", args=docker_args(self.container, spec),
                                       env=docker_env(spec, dict(os.environ)) or None))
        await client.__aenter__()
        return client

    async def _client(self, server_key: str) -> Client:
        lock = self._locks.setdefault(server_key, asyncio.Lock())
        async with lock:
            if server_key not in self._clients:
                if server_key not in self.specs:
                    raise DownstreamError(f"no launch config for server {server_key!r}")
                self._clients[server_key] = await self._new_client(self.specs[server_key])
                log.info("server started", extra={"ctx": {"server": server_key}})
            return self._clients[server_key]

    async def _drop(self, server_key: str) -> None:
        client = self._clients.pop(server_key, None)
        if client is not None:
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass
            log.warning("server connection dropped; next call restarts it", extra={"ctx": {"server": server_key}})

    async def call(self, server_key: str, tool_name: str, args: dict[str, Any]) -> str:
        """Call a tool; temporary overloads (HTTP 429/5xx from the tool's upstream API, also when reported
        inside a successful result) are retried after each delay in retry_delays_s."""
        for attempt, delay in enumerate((*self.retry_delays_s, None)):
            try:
                text = await self._call_once(server_key, tool_name, args)
                if delay is None or not TRANSIENT_IN_RESULT.search(text[:400]):
                    return text
            except DownstreamError as e:
                if delay is None or not TRANSIENT_ERROR.search(str(e)):
                    raise
            log.warning("temporary upstream overload; retrying", extra={"ctx": {
                "server": server_key, "tool": tool_name, "attempt": attempt + 1, "delay_s": delay}})
            await self._sleep(delay)
        raise AssertionError("unreachable")

    async def _call_once(self, server_key: str, tool_name: str, args: dict[str, Any]) -> str:
        client = await self._client(server_key)
        await self._pace(server_key)
        try:
            res = await client.call_tool(tool_name, args, timeout=self.timeout_s, raise_on_error=False)
        except Exception as e:
            # a server that crashed after starting would otherwise fail every later call in the run
            if re.search(r"connection closed|closedresource|broken pipe|brokenresource|end of file", str(e), re.I):
                await self._drop(server_key)
            raise
        text = "\n".join(getattr(b, "text", None) or json.dumps(b.model_dump(), default=str) for b in res.content)
        if res.is_error:
            raise DownstreamError(text[:2000])
        return text

    async def aclose(self) -> None:
        for key, client in list(self._clients.items()):
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                log.warning("server close failed", extra={"ctx": {"server": key}}, exc_info=True)
        self._clients.clear()

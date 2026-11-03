import asyncio
import json

from fastmcp import Client

from toolscope.gateway.downstream import docker_args
from toolscope.gateway.server import create_gateway, validate_args
from toolscope.retrievers.runlog import LOG_FIELDS

CARDS = {
    "fs/read_file": {"tool_id": "fs/read_file", "server_key": "fs", "tool_name": "read_file",
                     "description": "Read a file", "input_schema": {
                         "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    "web/fetch": {"tool_id": "web/fetch", "server_key": "web", "tool_name": "fetch",
                  "description": "Fetch a URL", "input_schema": {"type": "object", "properties": {}}},
}


class FakeRetriever:
    def __init__(self, ranking, fail=False):
        self.ranking, self.fail = ranking, fail

    def retrieve(self, query, k):
        if self.fail:
            raise RuntimeError("model crashed")
        return self.ranking[:k]


class FakePool:
    def __init__(self):
        self.calls = []

    async def call(self, server_key, tool_name, args):
        self.calls.append((server_key, tool_name, args))
        return "file contents"

    async def aclose(self):
        pass


def make(tmp_path, primary_fails=False):
    retrievers = {"toolret": FakeRetriever([("web/fetch", 0.9), ("fs/read_file", 0.5)], fail=primary_fails),
                  "bm25": FakeRetriever([("fs/read_file", 3.0), ("web/fetch", 1.0)])}
    pool = FakePool()
    gw = create_gateway(get_retriever=lambda name: retrievers[name], pool=pool, cards=CARDS,
                        strategy="toolret", qrels={"t1": {"fs/read_file": 1}},
                        log_dir=tmp_path, count_tokens=lambda s: len(s.split()), run_id="test")
    return gw, pool


def call(gw, tool, args):
    async def go():
        async with Client(gw) as c:
            return (await c.call_tool(tool, args, raise_on_error=False)).structured_content
    return asyncio.run(go())


def read_log(tmp_path, name):
    return [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]


def test_docker_args_pass_env_and_secret_placeholders_without_values():
    spec = {"command": "npx", "args": ["-y", "pkg"], "env": {"UV_EXCLUDE_NEWER": "2026", "KEY": "${KEY}"}}
    args = docker_args("ctr", spec)
    assert args[:2] == ["exec", "-i"]
    assert ["-e", "UV_EXCLUDE_NEWER=2026"] == args[2:4] and ["-e", "KEY"] == args[4:6]
    assert args[6] == "ctr" and args[-1].endswith('exec npx "-y" "pkg"')
    assert "${KEY}" not in " ".join(args)


def test_uv_offline_only_when_enabled_and_cacheable(monkeypatch):
    yahoo = {"command": "uvx", "args": ["mcp-yahoo-finance"]}
    assert "UV_OFFLINE=1" not in docker_args("ctr", yahoo)  # off by default
    monkeypatch.setenv("TOOLSCOPE_UV_OFFLINE", "1")
    assert docker_args("ctr", yahoo)[2:4] == ["-e", "UV_OFFLINE=1"]
    for spec in ({"command": "uvx", "args": ["-n", "mcp-music-analysis"]},
                 {"command": "uvx", "args": ["--from", "git+https://x/y.git", "y"]},
                 {"command": "uv", "args": ["run", "--with", "biomcp-python", "biomcp", "run"]},
                 {"command": "npx", "args": ["-y", "pkg"]}):
        assert "UV_OFFLINE=1" not in docker_args("ctr", spec)


def test_validate_args_reports_missing_required():
    assert validate_args(CARDS["fs/read_file"]["input_schema"], {}) != []
    assert validate_args(CARDS["fs/read_file"]["input_schema"], {"path": "/x"}) == []


def test_search_tools_returns_k_and_logs_one_line(tmp_path):
    gw, _ = make(tmp_path)
    out = call(gw, "search_tools", {"query": "read my file", "k": 1, "task_id": "t1"})
    assert [t["tool_id"] for t in out["result"]] == ["web/fetch"]
    rec = read_log(tmp_path, "gateway.jsonl")[0]
    assert list(rec)[:10] == LOG_FIELDS
    assert rec["strategy"] == "toolret" and rec["gold_rank"] == 2 and rec["fallback_used"] is False


def test_search_tools_falls_back_to_bm25_and_logs_it(tmp_path):
    gw, _ = make(tmp_path, primary_fails=True)
    out = call(gw, "search_tools", {"query": "read my file", "k": 1})
    assert out["result"][0]["tool_id"] == "fs/read_file"
    rec = read_log(tmp_path, "gateway.jsonl")[0]
    assert rec["fallback_used"] is True and rec["strategy"] == "bm25"


def test_call_tool_rejects_invalid_args_without_calling_server(tmp_path):
    gw, pool = make(tmp_path)
    out = call(gw, "call_tool", {"name": "fs/read_file", "args": {}})
    assert out["ok"] is False and out["args_valid"] is False and pool.calls == []
    assert read_log(tmp_path, "gateway_calls.jsonl")[0]["args_valid"] is False


def test_call_tool_unknown_tool(tmp_path):
    gw, pool = make(tmp_path)
    out = call(gw, "call_tool", {"name": "nope/x", "args": {}})
    assert out["ok"] is False and "unknown tool" in out["error"] and pool.calls == []


def test_call_tool_valid_runs_on_its_server(tmp_path):
    gw, pool = make(tmp_path)
    out = call(gw, "call_tool", {"name": "fs/read_file", "args": {"path": "/x"}})
    assert out["ok"] is True and out["result"] == "file contents"
    assert pool.calls == [("fs", "read_file", {"path": "/x"})]


def test_call_tool_selection_only_validates_and_logs_without_running(tmp_path):
    gw, pool = make(tmp_path)
    out = call(gw, "call_tool", {"name": "fs/read_file", "args": {"path": "/x"}, "execute": False})
    assert out["ok"] is True and out["args_valid"] is True and out["executed"] is False and pool.calls == []
    assert read_log(tmp_path, "gateway_calls.jsonl")[0]["executed"] is False


def test_search_results_expose_tool_id_as_the_only_identifier(tmp_path):
    gw, _ = make(tmp_path)
    out = call(gw, "search_tools", {"query": "read my file", "k": 2})
    assert set(out["result"][0]) == {"tool_id", "server", "description", "input_schema", "score"}


def test_docker_env_passes_only_placeholder_secrets_to_the_docker_process():
    from toolscope.gateway.downstream import docker_env
    spec = {"command": "npx", "args": [], "env": {"UV_EXCLUDE_NEWER": "2026", "KEY": "${KEY}"}}
    assert docker_env(spec, {"KEY": "s3cret", "OTHER": "x"}) == {"KEY": "s3cret"}
    assert docker_env({"command": "npx", "args": []}, {"KEY": "s3cret"}) == {}


def test_pool_drops_a_dead_connection_so_the_next_call_restarts_the_server():
    from toolscope.gateway.downstream import DownstreamPool

    class DeadThenAlive:
        def __init__(self, dead):
            self.dead, self.closed = dead, False

        async def call_tool(self, *a, **k):
            if self.dead:
                raise RuntimeError("Connection closed")
            class R:
                is_error, content = False, [type("B", (), {"text": "ok"})()]
            return R()

        async def __aexit__(self, *a):
            self.closed = True

    made = []

    class Pool(DownstreamPool):
        async def _new_client(self, spec):
            made.append(DeadThenAlive(dead=not made))
            return made[-1]

    async def go():
        pool = Pool("ctr", {"s": {"command": "x"}})
        try:
            await pool.call("s", "t", {})
        except Exception as e:
            assert "Connection closed" in str(e)
        return await pool.call("s", "t", {})

    assert asyncio.run(go()) == "ok" and len(made) == 2 and made[0].closed


def test_pool_spaces_calls_to_rate_limited_servers():
    from toolscope.gateway.downstream import DownstreamPool

    class Ok:
        async def call_tool(self, *a, **k):
            class R:
                is_error, content = False, [type("B", (), {"text": "ok"})()]
            return R()

        async def __aexit__(self, *a):
            pass

    waits = []

    class Pool(DownstreamPool):
        async def _new_client(self, spec):
            return Ok()

        async def _sleep(self, s):
            waits.append(round(s, 1))

    async def go():
        pool = Pool("ctr", {"a": {"command": "x"}, "b": {"command": "x"}}, min_interval_s={"a": 3.0, "b": 3.0},
                    rate_group={"a": "arxiv", "b": "arxiv"})
        pool._clock = iter([100.0, 100.0, 101.0, 101.0, 110.0, 110.0]).__next__
        await pool.call("a", "t", {})   # first call: no wait
        await pool.call("b", "t", {})   # same group, 1 s later: wait 2 s
        await pool.call("a", "t", {})   # 10 s later: no wait

    asyncio.run(go())
    assert waits == [2.0]


def test_pool_retries_temporary_overloads_then_succeeds():
    from toolscope.gateway.downstream import DownstreamPool, DownstreamError
    results = iter([("err", "Failed to find parking facilities: 502"),
                    ("ok", "Error: Page request resulted in HTTP 503 (https://export.arxiv.org/api)"),
                    ("ok", "parking: 3 results")])

    class Flaky:
        async def call_tool(self, *a, **k):
            kind, text = next(results)
            class R:
                is_error = kind == "err"
                content = [type("B", (), {"text": text})()]
            return R()

        async def __aexit__(self, *a):
            pass

    waits = []

    class Pool(DownstreamPool):
        async def _new_client(self, spec):
            return Flaky()

        async def _sleep(self, s):
            waits.append(s)

    out = asyncio.run(Pool("ctr", {"osm": {"command": "x"}}, retry_delays_s=(5, 20)).call("osm", "t", {}))
    assert out == "parking: 3 results" and waits == [5, 20]


def test_pool_does_not_retry_agent_errors():
    from toolscope.gateway.downstream import DownstreamPool, DownstreamError
    import pytest

    class Bad:
        calls = 0

        async def call_tool(self, *a, **k):
            Bad.calls += 1
            class R:
                is_error, content = True, [type("B", (), {"text": "sheet not found: Sheet1"})()]
            return R()

        async def __aexit__(self, *a):
            pass

    class Pool(DownstreamPool):
        async def _new_client(self, spec):
            return Bad()

        async def _sleep(self, s):
            pass

    with pytest.raises(DownstreamError):
        asyncio.run(Pool("ctr", {"x": {"command": "x"}}, retry_delays_s=(5, 20)).call("x", "t", {}))
    assert Bad.calls == 1


def test_call_tool_fails_fast_for_hosts_unreachable_from_this_network(tmp_path):
    gw, pool = make(tmp_path)
    out = call(gw, "call_tool", {"name": "fs/read_file", "args": {"path": "https://www.baidu.com/robots.txt"}})
    assert out["ok"] is False and "unreachable from the evaluation network" in out["error"] and pool.calls == []


def test_blocked_host_only_checks_address_arguments():
    from toolscope.gateway.server import blocked_host
    assert blocked_host({"url": "https://www.baidu.com/sitemap.xml"}) == "baidu.com"
    assert blocked_host({"domain": "baidu.com"}) == "baidu.com"
    assert blocked_host({"urls": ["https://example.com", "https://weibo.com/x"]}) == "weibo.com"
    assert blocked_host({"path": "/root/markdown/baidu.md", "content": "Report on www.baidu.com"}) is None

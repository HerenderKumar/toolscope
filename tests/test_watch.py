from toolscope.agent.watch import classify_error, summarize


def test_classify_environment_errors():
    for e in ["DownstreamError: Failed to initialize browser: Executable doesn't exist",
              "DownstreamError: connect ECONNREFUSED 1.2.3.4:443", "DownstreamError: Server disconnected",
              "DownstreamError: no such module: fts5", "McpError: Timed out while waiting for response",
              "DownstreamError: Failed to get nearby POIs: 429 Too Many Requests"]:
        assert classify_error(e) == "ENV", e


def test_classify_agent_errors():
    assert classify_error("invalid arguments: 'data' is a required property") == "AGENT"
    assert classify_error("unknown tool 'load'; use a tool_id returned by search_tools") == "AGENT"


def test_classify_unclear_is_check():
    assert classify_error("DownstreamError: Error executing tool: file not found /root/x.md") == "CHECK"


def test_summarize_flags_env_errors():
    rec = {"strategy": "bm25", "task_id": "abcdef1234", "latency_ms": 5000, "stop_reason": "final", "network_ok": True,
           "steps": [{"type": "call", "tool_id": "a/x", "ok": True, "error": None},
                     {"type": "call", "tool_id": "b/y", "ok": False, "error": "DownstreamError: Server disconnected"}]}
    line = summarize(rec, 3, {"abcdef1234": [{"acceptable": ["a/x"]}]})
    assert line.startswith("[3] bm25 abcdef12") and "ENV" in line and "b/y" in line and "gold=1" in line


def test_external_406_is_environment():
    assert classify_error("DownstreamError: Failed to find parking facilities: 406") == "ENV"


def test_new_lines_skips_already_reported(tmp_path):
    from toolscope.agent.watch import new_lines
    p = tmp_path / "a.jsonl"
    p.write_text('{"a":1}\n{"a":2}\n{"a":3}\n')
    assert [n for n, _ in new_lines(p, seen=2)] == [3]


def test_summarize_flags_soft_errors_inside_successful_results():
    rec = {"strategy": "bm25", "task_id": "abcdef1234", "latency_ms": 1000, "stop_reason": "final", "network_ok": True,
           "steps": [{"type": "call", "tool_id": "w/get_summary", "ok": True, "error": None,
                      "result_snippet": '{"summary":"Error retrieving summary for X: Expecting value"}'}]}
    assert "SOFT: w/get_summary" in summarize(rec, 1, {})


def test_soft_error_ignores_numbers_in_data_but_catches_status_codes():
    from toolscope.agent.watch import SOFT_PATTERNS
    assert not SOFT_PATTERNS.search("{'tempo': array([60.09]), 'beats': array([ 44, 87, 303, 518, 404])}")
    assert SOFT_PATTERNS.search("Error searching museum objects: Error: HTTP error! status: 410")
    assert SOFT_PATTERNS.search("(429, 'https://en.wikipedia.org/w/api.php')")
    assert SOFT_PATTERNS.search("Failed to retrieve cryptocurrency data")


def test_quiet_mode_reports_flagged_tasks_and_every_tenth():
    from toolscope.agent.watch import should_report
    assert should_report("[7] bm25 x | ENV: a", 7, quiet=True)
    assert not should_report("[7] bm25 x calls=1", 7, quiet=True)
    assert should_report("[10] bm25 x calls=1", 10, quiet=True)
    assert should_report("[7] bm25 x calls=1", 7, quiet=False)


def test_permanent_external_changes_are_ext_not_env():
    for e in ["DownstreamError: Error: DDG detected an anomaly in the request, you are likely making requests too quickly.",
              "DownstreamError: MCP error -32602: Input validation error: Invalid arguments for tool get-train-route-stations",
              "DownstreamError: Error searching museum objects: Error: HTTP error! status: 410"]:
        assert classify_error(e) == "EXT", e
    assert classify_error("DownstreamError: Failed to find parking facilities: 502") == "ENV"


def test_summarize_flags_ext():
    rec = {"strategy": "bm25", "task_id": "abcdef1234", "latency_ms": 1000, "stop_reason": "final", "network_ok": True,
           "steps": [{"type": "call", "tool_id": "d/s", "ok": False, "error": "Error: DDG detected an anomaly in the request"}]}
    assert "EXT: d/s" in summarize(rec, 1, {})


def test_live_server_missing_catalog_tool_is_ext_but_gateway_unknown_tool_is_agent():
    assert classify_error("McpError: Tool 'tft-recommend-champion-for-item' not found") == "EXT"
    assert classify_error("unknown tool 'a/b'; use a tool_id returned by search_tools") == "AGENT"


def test_soft_error_ignores_error_null_fields():
    from toolscope.agent.watch import SOFT_PATTERNS
    assert not SOFT_PATTERNS.search('{"status": "success", "error": null, "data": {"name": "apple"}}')
    assert SOFT_PATTERNS.search('{"status": "error", "message": "Paper not found"}')
    assert SOFT_PATTERNS.search('{"error": "No presentation is currently loaded"}')


def test_server_side_unknown_tool_is_ext():
    assert classify_error("DownstreamError: Unknown tool: excel_read") == "EXT"


def test_classify_call_uses_the_unreachable_source_list():
    from toolscope.agent.watch import classify_call
    assert classify_call("trends-hub/get-weibo-trending", 'DownstreamError: {"type": "text", "text": ""}') == "EXT"
    assert classify_call("trends-hub/get-36kr-trending", 'DownstreamError: {"type": "text", "text": ""}') == "CHECK"


def test_unreachable_host_error_is_ext():
    assert classify_error("Host unreachable from the evaluation network: www.baidu.com (D-041)") == "EXT"


def test_upstream_graphql_schema_change_is_ext():
    assert classify_error('DownstreamError: Error: API Error: 400 - {"errors":[{"message":"Cannot query field \\"asciiCode\\" on type"}]}') == "EXT"


def test_scraper_broken_by_site_change_is_ext():
    err = ('DownstreamError: {   "error": "Failed to parse search results from Airbnb. The page structure may have '
           'changed.",   "details": "Could not find data script element - page structure may have changed" }')
    assert classify_error(err) == "EXT"


def test_retired_datagov_api_is_ext():
    assert classify_error("DownstreamError: Data.gov API error: Request failed with status code 404\nStatus: 404") == "EXT"


def test_playwright_selector_timeout_is_agent():
    err = "DownstreamError: Operation failed: page.waitForSelector: Timeout 30000ms exceeded. Call log: ..."
    assert classify_error(err) == "AGENT"


def test_antv_remote_map_failure_is_ext():
    err = ("McpError: MCP error -32603: Failed to generate chart: Call tool generate_pin_map failed: "
           "Cannot read properties of undefined (reading 'length')")
    assert classify_error(err) == "EXT"


def test_soft_does_not_match_words_containing_exception():
    from toolscope.agent.watch import SOFT_PATTERNS
    assert not SOFT_PATTERNS.search("three exceptionally well-preserved Inca child mummies")
    assert SOFT_PATTERNS.search("Exception: boom")


def test_puppeteer_selector_timeout_is_agent():
    err = "DownstreamError: Failed to fill input[name=q]: Waiting for selector `input[name=q]` failed: Waiting failed: 30000ms exceeded"
    assert classify_error(err) == "AGENT"


def test_empty_youtube_transcript_is_flagged_and_ext():
    from toolscope.agent.watch import SOFT_PATTERNS
    res = '{"type": "text", "text": "", "metadata": {"videoId": "arj7oStGLkU", "language": "en", "charCount": 0}}'
    assert SOFT_PATTERNS.search(res)
    assert classify_error(res) == "EXT"
    assert not SOFT_PATTERNS.search('{"metadata": {"charCount": 0123}}'.replace("0123", "15230"))


def test_weread_unreachable_is_ext():
    from toolscope.agent.watch import classify_call
    assert classify_call("trends-hub/get-weread-rank", 'DownstreamError: {"type": "text", "text": ""}') == "EXT"


def test_cut_off_download_is_env():
    assert classify_error('Error: <urlopen error retrieval incomplete: got only 4194304 out of 10010725 bytes>') == "ENV"


def test_duckduckgo_rate_limit_is_ext_not_env():
    from toolscope.agent.watch import classify_call
    assert classify_call("duckduckgo-search/duckduckgo_web_search", "DownstreamError: Error: Rate limit exceeded") == "EXT"


def test_rerun_list_skips_blocked_duckduckgo():
    from toolscope.agent.status import rerun_list
    rec = {"strategy": "bm25", "task_id": "t1", "steps": [
        {"type": "call", "tool_id": "duckduckgo-search/duckduckgo_web_search", "error": "Error: Rate limit exceeded"}]}
    assert rerun_list([rec]) == []
    rec2 = {**rec, "task_id": "t2", "steps": [
        {"type": "call", "tool_id": "osm-mcp-server/find_nearby_places", "error": "Failed to get nearby POIs: 502"}]}
    assert rerun_list([rec2]) == [("bm25", "t2")]

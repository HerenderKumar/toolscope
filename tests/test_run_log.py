from toolscope.retrievers.runlog import LOG_FIELDS, gold_rank, make_record, schema_tokens

CARDS = {
    "a/x": {"tool_name": "x", "description": "does x", "input_schema": {"type": "object"}},
    "b/y": {"tool_name": "y", "description": "does y", "input_schema": {"type": "object"}},
}


def test_log_fields_are_the_ten_fixed_names():
    assert LOG_FIELDS == ["query_id", "task_id", "strategy", "k", "top_k", "gold_rank",
                          "verdict", "latency_ms", "tokens", "fallback_used"]


def test_gold_rank_is_best_one_based_rank_or_none():
    assert gold_rank(["c", "b", "a"], {"a", "b"}) == 2
    assert gold_rank(["c"], {"a"}) is None


def test_make_record_has_all_fields_plus_ts_and_run_id():
    full = [("b/y", 0.9), ("a/x", 0.5)]
    rec = make_record("run1", "bm25", "t1", 1, full, {"a/x"}, 12.345, 40)
    assert list(rec)[:10] == LOG_FIELDS and {"ts", "run_id"} <= set(rec)
    assert rec["top_k"] == [["b/y", 0.9]]
    assert rec["gold_rank"] == 2 and rec["verdict"] is None and rec["fallback_used"] is False
    assert rec["query_id"] == "run1/bm25/t1" and rec["latency_ms"] == 12.3 and rec["tokens"] == 40


def test_schema_tokens_counts_the_top_k_tool_schemas():
    n = schema_tokens(CARDS, ["a/x", "b/y"], lambda text: len(text.split()))
    assert n > schema_tokens(CARDS, ["a/x"], lambda text: len(text.split())) > 0

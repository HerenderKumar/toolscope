import json

from toolscope.agent.resume import completed_task_ids, wait_for_network

KEY = {"strategy": "bm25", "backend": "ollama", "model": "m", "mode": "execute", "k": 5}


def write(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_completed_task_ids_matches_config_and_skips_flagged(tmp_path):
    log = tmp_path / "agent.jsonl"
    write(log, [{**KEY, "task_id": "t1", "network_ok": True},
                {**KEY, "task_id": "t2", "network_ok": False},          # network dropped: redo
                {**KEY, "task_id": "t3"},                                # older record without flag: done
                {**KEY, "strategy": "semantic", "task_id": "t4", "network_ok": True},
                {**KEY, "task_id": "t5", "stop_reason": "error: boom"}])  # crashed: redo
    assert completed_task_ids(log, KEY) == {"t1", "t3"}


def test_completed_task_ids_missing_file(tmp_path):
    assert completed_task_ids(tmp_path / "none.jsonl", KEY) == set()


def test_wait_for_network_waits_until_up():
    states, sleeps = iter([False, False, True]), []
    waited = wait_for_network(lambda: next(states), sleep=sleeps.append, interval_s=30)
    assert waited == 60 and sleeps == [30, 30]


def test_wait_for_network_no_wait_when_up():
    assert wait_for_network(lambda: True, sleep=lambda s: None) == 0


def test_resume_key_includes_temperature_and_old_records_count_as_zero(tmp_path):
    log = tmp_path / "agent.jsonl"
    write(log, [{**KEY, "task_id": "t1"}])                       # temperature-0 era record, no field
    assert completed_task_ids(log, {**KEY, "temperature": 0.0}) == {"t1"}
    assert completed_task_ids(log, {**KEY, "temperature": 0.7}) == set()

"""Regression checks for the offline chart/case cohort and outcome precedence."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from failure_report import build_report, read_outcomes  # noqa: E402

MODEL = "deepseek/deepseek-v3.2"
AGENT = "Kimi K2.6"


def outcome(verdict, source="judge"):
    return {"verdict": verdict, "source": source, "exploratory": source == "judge"}


def row(task, label="agent selection error", cap="1"):
    return {"scope": "agent", "agent": AGENT, "strategy": "bm25", "task_id": task,
            "cap_id": task + "#" + cap, "label": label, "reason": "not_called"}


def test_human_override_latest_nonerror_judge_and_variant_isolation(tmp_path):
    judge = tmp_path / "judge.jsonl"
    human = tmp_path / "human.csv"
    records = [dict(agent=AGENT, strategy="bm25", task_id=t, model=MODEL, variant=v, verdict=verdict)
               for t, v, verdict in [("t1", "status", "success"), ("t2", "status", "success"),
                                      ("t2", "status", "failure"), ("t2", "status", "error"),
                                      ("t2", "original", "success")]]
    judge.write_text("".join(json.dumps(r) + "\n" for r in records))
    human.write_text("task_id,agent,strategy,label\nt1,Kimi K2.6,bm25,failure\n")
    outcomes = read_outcomes(judge, human, MODEL, "status")
    assert outcomes[(AGENT, "bm25", "t1")]["source"] == "human"
    assert outcomes[(AGENT, "bm25", "t1")]["verdict"] == "failure"
    assert outcomes[(AGENT, "bm25", "t1")]["evidence"]["record"] == 1
    assert outcomes[(AGENT, "bm25", "t2")]["verdict"] == "failure"
    assert outcomes[(AGENT, "bm25", "t2")]["evidence"]["line"] == 3
    assert outcomes[(AGENT, "bm25", "t2")]["exploratory"] is True
    assert read_outcomes(judge, human, MODEL, "original")[(AGENT, "bm25", "t2")]["verdict"] == "success"


def test_counts_come_from_membership_and_success_missing_and_solved_are_excluded():
    rows = [row("failed"), row("failed", cap="2"), row("success"), row("missing"), row("outside", "solved")]
    outcomes = {(AGENT, "bm25", t): outcome(v, src) for t, v, src in
                [("failed", "failure", "human"), ("success", "success", "human"), ("outside", "failure", "judge")]}
    report = build_report(rows, outcomes, [(AGENT, "bm25")], MODEL, "status", {})
    run = report["agent"][0]
    assert run["labels"]["agent selection error"] == 2
    ids = run["case_ids"]["agent selection error"]
    assert len(ids) == len(set(ids)) == 2
    assert {report["cases"][i]["task_id"] for i in ids} == {"failed"}
    assert run["failed_tasks"] == 2
    assert run["failed_tasks_with_labels"] == run["failed_tasks_without_labels"] == 1
    assert run["missing_outcomes"] == 1
    assert run["failed_by"] == "human + judge"
    assert report["report_id"] == build_report(list(reversed(rows)), outcomes, [(AGENT, "bm25")], MODEL,
                                               "status", {})["report_id"]
    assert report["report_id"] != build_report(rows, outcomes, [(AGENT, "bm25")], MODEL,
                                               "status", {"labels": "changed"})["report_id"]


def test_duplicate_capabilities_cannot_double_count_a_segment():
    with pytest.raises(ValueError, match="Duplicate capability"):
        build_report([row("t"), row("t")], {(AGENT, "bm25", "t"): outcome("failure")},
                     [(AGENT, "bm25")], MODEL, "status", {})


def test_validation_scope_is_limited_to_d075_model_variant_and_agents(tmp_path):
    judge = tmp_path / "judge.jsonl"
    records = [dict(agent=a, strategy="bm25", task_id="t", model=MODEL, variant=v, verdict="failure")
               for a in (AGENT, "Sonnet 5.5") for v in ("original", "status")]
    judge.write_text("".join(json.dumps(r) + "\n" for r in records))
    missing = tmp_path / "missing.csv"
    status = read_outcomes(judge, missing, MODEL, "status")
    assert not status[("Sonnet 5.5", "bm25", "t")]["exploratory"]
    assert status[(AGENT, "bm25", "t")]["exploratory"]
    assert read_outcomes(judge, missing, MODEL, "original")[("Sonnet 5.5", "bm25", "t")]["exploratory"]


def test_unparsed_response_preserves_legacy_count_but_discloses_raw_verdict(tmp_path):
    judge = tmp_path / "judge.jsonl"
    record = dict(agent=AGENT, strategy="bm25", task_id="t", model=MODEL, variant="status", verdict="unparsed")
    judge.write_text(json.dumps(record) + "\n")
    outcomes = read_outcomes(judge, tmp_path / "missing.csv", MODEL, "status")
    report = build_report([row("t")], outcomes, [(AGENT, "bm25")], MODEL, "status", {})
    assert report["agent"][0]["failed_tasks"] == 1
    saved = next(iter(report["cases"].values()))["outcome"]
    assert saved["verdict"] == "failure" and saved["raw_verdict"] == "unparsed"
    assert "Legacy report" in saved["interpretation"]


def test_v2_report_carries_evidence_and_refuses_legacy_labels():
    rows = [{**row("t"), "classifier_version": "D-078", "repair_assessment": {
        "status": "not_tested", "strategy": "bm25", "query_kind": "offline_task_question", "cutoff": 10}}]
    outcomes = {(AGENT, "bm25", "t"): outcome("failure", "human")}
    manifest = {"classifier_version": "D-078", "tested_repair_strategies": ["bm25", "semantic"]}
    report = build_report(rows, outcomes, [(AGENT, "bm25")], MODEL, "status", {}, "v2", manifest)
    assert report["classifier_version"] == "D-078" and report["diagnosis_version"] == "v2"
    assert next(iter(report["cases"].values()))["repair_assessment"]["status"] == "not_tested"
    with pytest.raises(ValueError, match="requires D-078"):
        build_report([row("t")], outcomes, [(AGENT, "bm25")], MODEL, "status", {}, "v2", manifest)


def test_v3_unassigned_is_coverage_not_a_seventh_fault_label():
    rows = [{**row(t, label), 'classifier_version': 'D-079', 'diagnostic_status': status}
            for t, label, status in [('unassigned', None, 'replay_mismatch'),
                                      ('assigned', 'retrieval miss', 'assigned'),
                                      ('solved', 'solved', 'acceptable_call_succeeded')]]
    outcomes = {(AGENT, 'bm25', r['task_id']): outcome('failure', 'human') for r in rows}
    report = build_report(rows, outcomes, [(AGENT, 'bm25')], MODEL, 'status', {}, 'v3', {'classifier_version': 'D-079'})
    run = report['agent'][0]
    assert run['total_labels'] == run['unassigned_capabilities'] == 1
    assert run['failed_tasks'] == 3
    assert run['failed_tasks_without_labels'] == 2
    assert run['failed_tasks_only_unassigned'] == run['failed_tasks_all_capabilities_solved'] == 1
    assert None not in run['case_ids'] and 'unassigned' not in run['labels']
    assert report['cases'][run['unassigned_case_ids'][0]]['label'] is None
    assert run['diagnostic_status_counts'] == {'replay_mismatch': 1}
    with pytest.raises(ValueError, match='v3 requires'):
        build_report([row('t')], outcomes, [(AGENT, 'bm25')], MODEL, 'status', {}, 'v3', {'classifier_version': 'D-079'})

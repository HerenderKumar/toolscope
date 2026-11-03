import json

import pytest
from fastapi.testclient import TestClient

from toolscope.api import app as api


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) if isinstance(rows, list) else json.dumps(rows))


def _setup(tmp_path, monkeypatch):
    card = {"tool_id": "s/a", "server_name": "S", "server_key": "s", "tool_name": "a", "description": "old desc",
            "params": [], "search_text": "S | a | old desc"}
    task = {"task_id": "t1", "question": "find a", "capabilities": [{"cap_id": "t1#1", "annotated_name": "a",
                                                                     "acceptable": ["s/a"]}]}
    _write(tmp_path / "data/catalog/tools.jsonl", [card])
    _write(tmp_path / "data/gold/gold.jsonl", [task])
    _write(tmp_path / "results/metrics.json", {"baseline": "bm25", "strategies": {}})
    _write(tmp_path / "results/tool_health.jsonl", [{"tool_id": "s/a", "flags": [], "missed": 1}])
    _write(tmp_path / "results/failure_labels.jsonl",
           [{"scope": "agent", "agent": "Kimi K2.6", "strategy": "bm25", "task_id": "t1", "cap_id": "t1#1",
             "label": "retrieval miss", "reason": "whatif_no_fix"}])
    case = {"scope": "agent", "agent": "Kimi K2.6", "strategy": "bm25", "task_id": "t1", "cap_id": "t1#1",
            "label": "retrieval miss", "reason": "whatif_no_fix", "case_id": "case1",
            "outcome": {"verdict": "failure", "source": "human", "exploratory": False}}
    report = {"schema_version": 2, "report_id": "snapshot1", "judge_variant": "status",
              "outcome_policy": "D-075/D-076", "cases": {"case1": case},
              "agent": [{"agent": "Kimi K2.6", "strategy": "bm25", "failed_by": "human", "exploratory": False,
                         "labels": {"retrieval miss": 1, "composition error": 0},
                         "case_ids": {"retrieval miss": ["case1"], "composition error": []}}]}
    _write(tmp_path / "results/failures_status.json", report)
    monkeypatch.setattr(api, "ROOT", tmp_path)
    monkeypatch.setattr(api, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(api, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(api, "CATALOG", tmp_path / "data/catalog/tools.jsonl")
    monkeypatch.setattr(api, "GOLD", tmp_path / "data/gold/gold.jsonl")
    for f in (api._cards, api._gold, api._editor):
        f.cache_clear()


def test_get_endpoints_serve_files_only(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    c = TestClient(api.app)
    assert c.get("/api/metrics").json()["baseline"] == "bm25"
    assert c.get("/api/tools").json()[0]["tool_id"] == "s/a"
    case = c.get("/api/cases", params={"agent": "Kimi K2.6", "strategy": "bm25", "label": "retrieval miss"}).json()
    assert case[0]["capability"] == "a" and case[0]["acceptable"] == ["s/a"]
    assert c.get("/api/tools/s/a").json()["tasks"][0]["task_id"] == "t1"
    assert c.get("/api/task_success").status_code == 404  # missing results file -> clear 404
    assert c.get("/api/judge", params={"variant": "nope"}).status_code == 400


def test_cases_page_uses_exported_membership_not_raw_labels(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    path = tmp_path / "results/failures_status.json"
    report = json.loads(path.read_text())
    # Three distinct capability rows in the chart; the raw labels file has only one.
    for i in (2, 3):
        report["cases"][f"case{i}"] = {**report["cases"]["case1"], "case_id": f"case{i}", "cap_id": f"t1#{i}"}
    task = json.loads((tmp_path / "data/gold/gold.jsonl").read_text())
    task["capabilities"] += [{**task["capabilities"][0], "cap_id": f"t1#{i}"} for i in (2, 3)]
    _write(tmp_path / "data/gold/gold.jsonl", [task])
    report["agent"][0]["case_ids"]["retrieval miss"] = ["case1", "case2", "case3"]
    report["agent"][0]["labels"]["retrieval miss"] = 3
    _write(path, report)
    params = {"agent": "Kimi K2.6", "strategy": "bm25", "label": "retrieval miss", "paginated": True,
              "limit": 2, "report_id": "snapshot1"}
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    client = TestClient(api.app)
    first = client.get("/api/cases", params=params).json()
    second = client.get("/api/cases", params={**params, "offset": 2}).json()
    assert first["total"] == second["total"] == 3
    assert [r["case_id"] for r in first["items"] + second["items"]] == ["case1", "case2", "case3"]
    assert first["items"][0]["outcome"]["source"] == "human"
    assert client.get("/api/cases", params={**params, "offset": 3}).json()["items"] == []
    empty = client.get("/api/cases", params={**params, "label": "composition error"}).json()
    assert empty["total"] == 0 and empty["items"] == []
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("extra,status", [({"limit": 0}, 422), ({"limit": -1}, 422), ({"limit": 501}, 422),
                                        ({"offset": -1}, 422), ({"variant": "nope"}, 400),
                                        ({"label": "typo"}, 400), ({"agent": "unknown"}, 404),
                                        ({"report_id": "old-snapshot"}, 409)])
def test_case_pagination_rejects_invalid_inputs(tmp_path, monkeypatch, extra, status):
    _setup(tmp_path, monkeypatch)
    params = {"agent": "Kimi K2.6", "strategy": "bm25", "label": "retrieval miss", **extra}
    assert TestClient(api.app).get("/api/cases", params=params).status_code == status


def test_original_variant_is_separate_and_old_reports_require_export(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    report = json.loads((tmp_path / "results/failures_status.json").read_text())
    report.update(report_id="original-snapshot", judge_variant="original")
    report["agent"][0]["labels"]["retrieval miss"] = 0
    report["agent"][0]["case_ids"]["retrieval miss"] = []
    _write(tmp_path / "results/failures.json", report)
    params = {"agent": "Kimi K2.6", "strategy": "bm25", "label": "retrieval miss", "paginated": True}
    client = TestClient(api.app)
    assert client.get("/api/cases", params={**params, "variant": "original"}).json()["total"] == 0
    assert client.get("/api/cases", params=params).json()["total"] == 1
    _write(tmp_path / "results/failures_status.json", {"agent": []})
    response = client.get("/api/cases", params=params)
    assert response.status_code == 409 and "failure_report.py" in response.json()["detail"]


def test_diagnosis_versions_and_repair_evidence_are_served_without_mixing(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    report = json.loads((tmp_path / "results/failures_status.json").read_text())
    report.update(report_id="v2-snapshot", diagnosis_version="v2", classifier_version="D-078")
    report["cases"]["case1"]["repair_assessment"] = {"status": "not_tested"}
    _write(tmp_path / "results/diagnosis_v2/failures_status.json", report)
    evidence = [{"task_id": "t1", "strategy": "bm25", "cap_id": "t1#1", "before_rank": 40, "after_rank": 2},
                {"task_id": "t2", "strategy": "bm25", "cap_id": "t2#1", "before_rank": 20, "after_rank": 15}]
    _write(tmp_path / "results/diagnosis_v2/repair_evidence.jsonl", evidence)
    client = TestClient(api.app)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert client.get("/api/failures").json()["report_id"] == "snapshot1"
    assert client.get("/api/failures", params={"diagnosis": "v2"}).json()["classifier_version"] == "D-078"
    params = {"agent": "Kimi K2.6", "strategy": "bm25", "label": "retrieval miss", "paginated": True,
              "diagnosis": "v2", "report_id": "v2-snapshot"}
    response = client.get("/api/cases", params=params).json()
    assert response["diagnosis_version"] == "v2"
    assert response["items"][0]["repair_assessment"]["status"] == "not_tested"
    assert client.get("/api/cases", params={**params, "report_id": "snapshot1"}).status_code == 409
    for path in ("/api/failures", "/api/cases"):
        assert client.get(path, params={**params, "diagnosis": "../../data"}).status_code == 400
    assert client.get("/api/repair_evidence", params={"task_id": "t1", "strategy": "bm25"}).json() == evidence[:1]
    assert client.get("/api/repair_evidence", params={"task_id": "t1", "strategy": "toolret"}).json() == []
    assert {p: p.read_bytes() for p in before} == before


def test_whatif_writes_nothing(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)

    class FakeEditor:
        def whatif(self, tool_id, description, queries):
            return [{"query": q, "bm25": [5, 1], "semantic": [3, 2]} for q in queries]

    monkeypatch.setattr(api, "_editor", lambda: FakeEditor())
    before = {p: p.read_text() for p in tmp_path.rglob("*") if p.is_file()}
    r = TestClient(api.app).post("/api/whatif", json={"tool_id": "s/a", "description": "new", "task_ids": ["t1"]})
    assert r.json()["results"][0]["bm25"] == [5, 1]
    assert {p: p.read_text() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_v3_unassigned_cases_are_separate_readonly_and_snapshot_bound(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    report = json.loads((tmp_path / 'results/failures_status.json').read_text())
    report.update(report_id='v3-snapshot', diagnosis_version='v3', classifier_version='D-079')
    case = report['cases']['case1']
    case.update(label=None, diagnostic_status='replay_mismatch', search_evidence={'events': []},
                offline_diagnosis={'label': 'bad description'})
    run = report['agent'][0]
    run.update(unassigned_capabilities=1, unassigned_case_ids=['case1'])
    run['labels']['retrieval miss'] = 0
    run['case_ids']['retrieval miss'] = []
    _write(tmp_path / 'results/diagnosis_v3/failures_status.json', report)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    client = TestClient(api.app)
    params = {'agent': 'Kimi K2.6', 'strategy': 'bm25', 'label': 'unassigned', 'paginated': True,
              'diagnosis': 'v3', 'report_id': 'v3-snapshot'}
    assert client.get('/api/failures', params={'diagnosis': 'v3'}).json()['report_id'] == 'v3-snapshot'
    data = client.get('/api/cases', params=params).json()
    assert data['total'] == 1 and data['items'][0]['label'] is None
    assert data['items'][0]['search_evidence'] == {'events': []}
    assert client.get('/api/cases', params={**params, 'offset': 1}).json()['items'] == []
    assert client.get('/api/cases', params={**params, 'label': 'retrieval miss'}).json()['total'] == 0
    assert client.get('/api/cases', params={**params, 'report_id': 'snapshot1'}).status_code == 409
    assert client.get('/api/cases', params={**params, 'diagnosis': 'v1', 'report_id': 'snapshot1'}).status_code == 400
    assert {p: p.read_bytes() for p in before} == before

"""Read-only API for the ToolScope workbench (Phase 6). Every GET serves numbers that already exist in results/ or
logs/ (CLAUDE.md); the catalog and the frozen answer key are read for names, descriptions and gold tools.
POST /api/whatif computes ranks on in-memory copies and writes nothing.

Run:  PYTHONPATH=src .venv/bin/uvicorn toolscope.api.app:app --port 8000
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from toolscope.diagnosis.runs import RUNS, load

ROOT = Path(os.environ.get("TOOLSCOPE_ROOT", "."))
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
CATALOG = ROOT / os.environ.get("TOOLSCOPE_CATALOG", "data/catalog/tools.jsonl")
GOLD = ROOT / "data/gold/gold.jsonl"
VARIANTS = {"original": "", "status": "_status"}

app = FastAPI(title="ToolScope API", description="Read-only workbench API")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("TOOLSCOPE_CORS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


def _json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(404, f"{path.relative_to(ROOT)} not found; run the pipeline script that writes it")
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise HTTPException(404, f"{path.relative_to(ROOT)} not found")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _suffix(variant: str) -> str:
    if variant not in VARIANTS:
        raise HTTPException(400, f"variant must be one of {sorted(VARIANTS)}")
    return VARIANTS[variant]


def _diagnosis_dir(diagnosis: str) -> Path:
    if diagnosis not in {"v1", "v2", "v3"}:
        raise HTTPException(400, "diagnosis must be v1, v2 or v3")
    return RESULTS / f"diagnosis_{diagnosis}" if diagnosis != "v1" else RESULTS


@lru_cache(maxsize=1)
def _cards() -> dict[str, dict[str, Any]]:
    return {c["tool_id"]: c for c in _jsonl(CATALOG)}


@lru_cache(maxsize=1)
def _gold() -> dict[str, dict[str, Any]]:
    return {t["task_id"]: t for t in _jsonl(GOLD)}


@lru_cache(maxsize=None)
def _run(agent: str, strategy: str) -> dict[str, dict[str, Any]]:
    if (agent, strategy) not in RUNS:
        raise HTTPException(404, f"unknown run {agent}/{strategy}")
    return load(agent, strategy, RUNS[(agent, strategy)], root=ROOT)


@lru_cache(maxsize=1)
def _retrieval() -> dict[tuple[str, str], list[list]]:
    path = LOGS / "retrieval_k20.jsonl"
    path = path if path.exists() else LOGS / "retrieval.jsonl"
    return {(r["strategy"], r["task_id"]): r["top_k"] for r in _jsonl(path)}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics() -> Any:
    """Retrieval metrics per strategy (results/metrics.json = results/table.md)."""
    return _json(RESULTS / "metrics.json")


@app.get("/api/agent_runs")
def agent_runs() -> Any:
    return _json(RESULTS / "agent_runs.json")


@app.get("/api/task_success")
def task_success(variant: str = "status") -> Any:
    return _json(RESULTS / f"task_success{_suffix(variant)}.json")


@app.get("/api/judge")
def judge(variant: str = "status") -> Any:
    return _json(RESULTS / f"judge_agreement{_suffix(variant)}.json")


@app.get("/api/failures")
def failures(variant: str = "status", diagnosis: str = "v1") -> Any:
    return _json(_diagnosis_dir(diagnosis) / f"failures{_suffix(variant)}.json")


@app.get("/api/runs")
def runs() -> list[dict[str, str]]:
    return [{"agent": a, "strategy": s} for a, s in RUNS]


@app.get("/api/cases")
def cases(agent: str, strategy: str, label: str, limit: int = Query(50, ge=1, le=500),
          offset: int = Query(0, ge=0), variant: str = "status", paginated: bool = False,
          report_id: str | None = None, diagnosis: str = "v1") -> Any:
    """Serve exact chart members; paginated=true adds the saved total and snapshot.

    The default remains an array for existing clients. No outcomes or counts are
    recomputed here. A chart may supply report_id to reject a changed snapshot.
    """
    report = _json(_diagnosis_dir(diagnosis) / f"failures{_suffix(variant)}.json")
    if report.get("schema_version") != 2:
        raise HTTPException(409, "Case membership is not exported. Run scripts/failure_report.py "
                            f"with TOOLSCOPE_JUDGE_VARIANT={variant}.")
    if report_id is not None and report_id != report["report_id"]:
        raise HTTPException(409, "Failure report changed. Reload the chart before opening its cases.")
    run = next((r for r in report["agent"] if r["agent"] == agent and r["strategy"] == strategy), None)
    if run is None:
        raise HTTPException(404, f"unknown run {agent}/{strategy}")
    if label == "unassigned" and diagnosis == "v3":
        # A separate coverage list, never a seventh fault label or a chart segment.
        case_ids = run["unassigned_case_ids"]
        total = run["unassigned_capabilities"]
    elif label in run["case_ids"]:
        case_ids = run["case_ids"][label]
        total = run["labels"][label]
    else:
        raise HTTPException(400, "unknown failure label")
    gold = _gold()
    out = []
    for case_id in case_ids[offset:offset + limit]:
        r = report["cases"][case_id]
        cap = next(c for c in gold[r["task_id"]]["capabilities"] if c["cap_id"] == r["cap_id"])
        out.append({**r, "question": gold[r["task_id"]]["question"], "capability": cap["annotated_name"],
                    "acceptable": cap["acceptable"]})
    if paginated:
        return {"items": out, "total": total, "offset": offset, "limit": limit,
                "report_id": report["report_id"], "judge_variant": report["judge_variant"],
                "outcome_policy": report["outcome_policy"], "failed_by": run["failed_by"],
                "exploratory": run["exploratory"], "diagnosis_version": diagnosis,
                "classifier_version": report.get("classifier_version")}
    return out


@app.get("/api/repair_evidence")
def repair_evidence(task_id: str, strategy: str | None = None, cap_id: str | None = None) -> Any:
    """Saved D-078 observations only; an empty list is not proof of no repair effect."""
    return [r for r in _jsonl(RESULTS / "diagnosis_v2/repair_evidence.jsonl")
            if r["task_id"] == task_id and (strategy is None or r["strategy"] == strategy)
            and (cap_id is None or r["cap_id"] == cap_id)]


@app.get("/api/trajectory")
def trajectory(agent: str, strategy: str, task_id: str) -> dict[str, Any]:
    """One agent record: searches with shortlists, calls with status, final answer (logs only)."""
    rec = _run(agent, strategy).get(task_id)
    if rec is None:
        raise HTTPException(404, "no record for this task in this run")
    steps = []
    for s in rec["steps"]:
        if s.get("type") == "search":
            steps.append({"type": "search", "query": s.get("query"), "shortlist": s.get("shortlist", [])})
        elif s.get("type") == "call":
            steps.append({"type": "call", "tool_id": s["tool_id"], "args": s.get("args"), "ok": s.get("ok"),
                          "error": s.get("error"), "result_snippet": s.get("result_snippet")})
    return {"task_id": task_id, "agent": agent, "strategy": strategy, "stop_reason": rec.get("stop_reason"),
            "final_answer": rec.get("final_answer"), "steps": steps}


@app.get("/api/queries/{task_id}")
def query(task_id: str) -> dict[str, Any]:
    """The task question, its gold capabilities and every strategy's top-10 (logs/retrieval*.jsonl)."""
    task = _gold().get(task_id)
    if task is None:
        raise HTTPException(404, "unknown task")
    per_query = [r for r in _jsonl(RESULTS / "per_query.jsonl") if r["task_id"] == task_id]
    tops = {s: [t for t, _ in top[:10]] for (s, tid), top in _retrieval().items() if tid == task_id}
    return {"task_id": task_id, "question": task["question"], "capabilities": task["capabilities"],
            "top10": tops, "metrics": per_query}


@app.get("/api/tools")
def tools(limit: int = Query(100, le=600), flag: str | None = None) -> list[dict[str, Any]]:
    rows = _jsonl(RESULTS / "tool_health.jsonl")
    if flag:
        rows = [r for r in rows if flag in r["flags"]]
    return rows[:limit]


@app.get("/api/tools/{tool_id:path}")
def tool(tool_id: str) -> dict[str, Any]:
    card = _cards().get(tool_id)
    if card is None:
        raise HTTPException(404, "unknown tool")
    health = next((r for r in _jsonl(RESULTS / "tool_health.jsonl") if r["tool_id"] == tool_id), None)
    rewrites = [r for r in _jsonl(RESULTS / "rewrites.jsonl") if r["tool_id"] == tool_id] \
        if (RESULTS / "rewrites.jsonl").exists() else []
    whatif = [r for r in _jsonl(RESULTS / "whatif.jsonl") if r["tool_id"] == tool_id] \
        if (RESULTS / "whatif.jsonl").exists() else []
    tasks = [{"task_id": t["task_id"], "question": t["question"]} for t in _gold().values()
             if any(tool_id in c["acceptable"] for c in t["capabilities"])]
    return {"card": {k: card[k] for k in ("tool_id", "server_name", "tool_name", "description", "params")},
            "health": health, "rewrite": rewrites[-1] if rewrites else None, "whatif": whatif, "tasks": tasks}


class WhatIfRequest(BaseModel):
    tool_id: str
    description: str = Field(min_length=1, max_length=4000)
    task_ids: list[str] = Field(min_length=1, max_length=20)


@lru_cache(maxsize=1)
def _editor():
    from toolscope.diagnosis.whatif import WhatIf  # loads the semantic encoder once, on first use
    return WhatIf(list(_cards().values()))


@app.post("/api/whatif")
def whatif(req: WhatIfRequest) -> dict[str, Any]:
    """Rank of the tool for each task question before/after the new description (bm25, semantic). Writes nothing."""
    if req.tool_id not in _cards():
        raise HTTPException(404, "unknown tool")
    gold = _gold()
    missing = [t for t in req.task_ids if t not in gold]
    if missing:
        raise HTTPException(404, f"unknown task ids: {missing}")
    res = _editor().whatif(req.tool_id, req.description, [gold[t]["question"] for t in req.task_ids])
    return {"tool_id": req.tool_id, "results": [{"task_id": t, **r} for t, r in zip(req.task_ids, res)]}

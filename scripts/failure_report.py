"""Export failure counts and their exact case membership from saved evidence.

D-075/D-076 task outcomes use human labels
where available, otherwise the selected judge. Schema 2 adds stable case IDs,
outcome provenance and a snapshot ID so the chart and drill-down share a cohort.
All counting happens here, offline; the API only serves the exported results.

Run: TOOLSCOPE_JUDGE_VARIANT=status PYTHONPATH=src .venv/bin/python scripts/failure_report.py
Add --diagnosis v2 for D-078, or v3 for D-079 actual agent searches. Outputs are versioned; v1 remains the default.
"""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from toolscope.diagnosis.classify import LABELS
from toolscope.diagnosis.runs import RUNS

LABELS_PATH = Path("results/failure_labels.jsonl")
JUDGE = Path(os.environ.get("TOOLSCOPE_JUDGE_LOG", "logs/judge.jsonl"))
MODEL = os.environ.get("TOOLSCOPE_JUDGE_MODEL", "deepseek/deepseek-v3.2")
VARIANT = os.environ.get("TOOLSCOPE_JUDGE_VARIANT", "original")
HUMAN = Path("data/human/human_labels.csv")
OUT = Path("results/failures.md" if VARIANT == "original" else f"results/failures_{VARIANT}.md")
SHORT = {"environment failure": "Environment", "agent selection error": "Agent selection",
         "composition error": "Composition", "ranking confusion": "Ranking confusion",
         "bad description": "Bad description", "retrieval miss": "Retrieval miss"}
RunKey = tuple[str, str]
OutcomeKey = tuple[str, str, str]


def read_outcomes(judge_path: Path, human_path: Path, model: str, variant: str) -> dict[OutcomeKey, dict[str, Any]]:
    """Latest non-error judge verdict per task, overridden by the author's labels."""
    outcomes = {}
    for line_number, line in enumerate(judge_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["model"] != model or r.get("variant", "original") != variant or r["verdict"] == "error":
            continue
        covered = (model == "deepseek/deepseek-v3.2" and variant == "status"
                   and r["agent"] in {"Qwen2.5-7B (local)", "Qwen3-235B", "Sonnet 5.5"})
        outcomes[(r["agent"], r["strategy"], r["task_id"])] = {
            # Preserve the existing report's non-success/non-error handling, including
            # unparsed responses. Expose the raw value rather than implying a parsed failure.
            "verdict": "success" if r["verdict"] == "success" else "failure",
            "raw_verdict": r["verdict"], "source": "judge", "model": model, "variant": variant,
            "interpretation": ("Parsed judge verdict" if r["verdict"] in {"success", "failure"}
                               else "Legacy report counts this non-success, non-error response as failure"),
            "validation_scope": ("D-075: mixed held-out sample of 60; not separate subgroup validation"
                                 if covered else "Outside D-075 validated reporting scope"),
            "exploratory": not covered, "evidence": {"path": str(judge_path), "line": line_number},
        }
    if human_path.exists():
        with human_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for record_number, r in enumerate(reader, 1):
                if r["label"] not in {"success", "failure"}:
                    raise ValueError(f"Invalid human label for {r['task_id']}")
                outcomes[(r["agent"], r["strategy"], r["task_id"])] = {
                    "verdict": r["label"], "raw_verdict": r["label"], "source": "human", "model": None,
                    "variant": None, "interpretation": "Author's label overrides the judge",
                    "validation_scope": "Author's task-success label (D-075)", "exploratory": False,
                    "evidence": {"path": str(human_path), "record": record_number},
                }
    return outcomes


def build_report(rows: list[dict[str, Any]], outcomes: dict[OutcomeKey, dict[str, Any]],
                 runs: list[RunKey], model: str, variant: str,
                 source_sha256: dict[str, str], diagnosis: str = "v1",
                 repair_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build counts from the same saved membership consumed by /api/cases."""
    if diagnosis not in {"v1", "v2", "v3"}:
        raise ValueError("diagnosis must be v1, v2 or v3")
    if diagnosis == "v2" and (not repair_manifest or any(r.get("classifier_version") != "D-078" for r in rows)):
        raise ValueError("v2 requires D-078 labels and its repair manifest")
    if diagnosis == "v3" and (not repair_manifest or any(r.get("classifier_version") !=
            ("D-079" if r["scope"] == "agent" else "D-078") for r in rows)):
        raise ValueError("v3 requires D-079 agent labels, D-078 offline labels and its manifest")
    data = {"schema_version": 2, "classifier_version": "D-079 (agents) / D-078 (offline)" if diagnosis == "v3" else
            "D-078" if diagnosis == "v2" else "D-010/D-073 (unchanged)",
            "outcome_policy": "D-075/D-076: human overrides selected judge; unvalidated outcomes are exploratory",
            "judge_variant": variant, "judge_model": model, "source_sha256": source_sha256,
            "agent": [], "retrieval": [], "cases": {}}
    if diagnosis in {"v2", "v3"}:
        data.update(diagnosis_version=diagnosis, repair_scope=repair_manifest)
    for agent, strategy in runs:
        run = [r for r in rows if r["scope"] == "agent" and r["agent"] == agent and r["strategy"] == strategy]
        tasks = {r["task_id"] for r in run}
        selected = {t: outcomes[(agent, strategy, t)] for t in tasks if (agent, strategy, t) in outcomes}
        failed = {t for t, o in selected.items() if o["verdict"] == "failure"}
        members = {label: [] for label in LABELS}
        unassigned = []
        for r in sorted(run, key=lambda r: (r["task_id"], r["cap_id"])):
            if r["task_id"] not in failed or r["label"] == "solved":
                continue
            if r["label"] not in members and not (diagnosis == "v3" and r["label"] is None):
                raise ValueError(f"Unknown failure label: {r['label']}")
            identity = [agent, strategy, r["task_id"], r["cap_id"]]
            case_id = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
            if case_id in data["cases"]:
                raise ValueError(f"Duplicate capability: {identity}")
            data["cases"][case_id] = {**r, "case_id": case_id, "outcome": selected[r["task_id"]]}
            if r["label"] is None:
                unassigned.append(case_id)
            else:
                members[r["label"]].append(case_id)
        sources = {o["source"] for o in selected.values()}
        failed_by = ("human" if sources == {"human"} else "human + judge" if sources == {"human", "judge"}
                     else "judge (not validated for Kimi, D-075)" if agent.startswith("Kimi") else "judge")
        covered = {data["cases"][cid]["task_id"] for ids in members.values() for cid in ids}
        data["agent"].append({
            "agent": agent, "strategy": strategy, "failed_tasks": len(failed),
            "labels": {label: len(ids) for label, ids in members.items()}, "case_ids": members,
            "total_labels": sum(map(len, members.values())), "failed_by": failed_by,
            "outcome_sources": dict(Counter(o["source"] for o in selected.values())),
            "exploratory": any(o["exploratory"] for o in selected.values()),
            "missing_outcomes": len(tasks - selected.keys()),
            "failed_tasks_with_labels": len(covered), "failed_tasks_without_labels": len(failed - covered),
        })
        if diagnosis == "v3":
            unassigned_tasks = {data["cases"][cid]["task_id"] for cid in unassigned}
            data["agent"][-1].update(failed_tasks_with_unassigned=len(unassigned_tasks),
                                     failed_tasks_only_unassigned=len(unassigned_tasks - covered),
                                     failed_tasks_all_capabilities_solved=len(failed - covered - unassigned_tasks),
                                     unassigned_capabilities=len(unassigned),
                                     unassigned_case_ids=unassigned,
                                     diagnostic_status_counts=dict(Counter(data["cases"][cid]["diagnostic_status"] for cid in unassigned)))
    for strategy in dict.fromkeys(r["strategy"] for r in rows if r["scope"] == "retrieval"):
        c = Counter(r["label"] for r in rows if r["scope"] == "retrieval" and r["strategy"] == strategy)
        data["retrieval"].append({"strategy": strategy, "labels": dict(c)})
    data["report_id"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return data


def render_markdown(data: dict[str, Any]) -> str:
    """Preserve the existing paper table and its counts."""
    head = "| " + " | ".join(SHORT[x] for x in LABELS) + " |"
    sep = "|---" * (len(LABELS) + 3) + "|"
    lines = ["## Whose fault? Failure breakdown", "",
             "Agent runs: one label per unsolved capability (D-010) in **failed** tasks (human labels where they exist, "
             "otherwise the judge; the judge is not validated for Kimi, D-075); share of all such labels in brackets.", "",
             "| Agent | Strategy | Failed tasks " + head, sep]
    for r in data["agent"]:
        c = r["labels"]
        total = r["total_labels"] or 1
        cells = " | ".join(f"{c[x]} ({c[x] / total:.0%})" for x in LABELS)
        lines.append(f"| {r['agent']} | {r['strategy']} | {r['failed_tasks']} | {cells} |")
    if data.get("diagnosis_version") == "v3":
        lines[2] = ("Agent runs: D-079 labels for unsolved capabilities with sufficient recorded evidence in failed tasks. "
                    "Shares use assigned labels only; unassigned capabilities are reported separately below. "
                    "Human labels override the judge; Kimi semantic outcomes remain exploratory (D-075).")
    lines += ["", "Retrieval only (task question, top-10; 255 capabilities per strategy):", "",
              "| Strategy | Found | Ranking confusion | Bad description | Retrieval miss |", "|---|---|---|---|---|"]
    for r in data["retrieval"]:
        c = Counter(r["labels"])
        lines.append(f"| {r['strategy']} | {c['found']} | {c['ranking confusion']} | {c['bad description']} "
                     f"| {c['retrieval miss']} |")
    if data.get("diagnosis_version") == "v2":
        lines += ["", "D-078: description-repairable labels require a same-strategy rescue on the original task "
                  "question at top-10. BM25 and semantic were tested; ToolRet, hybrid, rerank and server-first "
                  "were not tested for repair. A zero in their description column is not evidence of no effect.",
                  "Agent rows use an offline task-question proxy; actual agent searches/top-5 were not replayed.",
                  "Legacy D-073 probe linkage is reconstructed from saved inputs, not newly measured."]
    if data.get("diagnosis_version") == "v3":
        lines += ["", "D-079: agent rows use observed query/cutoff and chronological exposure; offline rows retain D-078.",
                  "Unassigned capabilities are coverage limits, not a seventh category. Actual-query description repair and dependency failures were not tested.",
                  "Offline repair probes exist for BM25 and semantic only; zero description labels for the other four retrievers mean not tested.",
                  "Recorded event order is available; assistant-turn grouping is not. Query adequacy is not judged.",
                  "", "| Agent | Strategy | Unassigned capabilities in failed tasks |", "|---|---|---:|"]
        lines += [f"| {r['agent']} | {r['strategy']} | {r['unassigned_capabilities']} |" for r in data["agent"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", choices=("v1", "v2", "v3"), default="v1")
    diagnosis = parser.parse_args().diagnosis
    folder = Path(f"results/diagnosis_{diagnosis}") if diagnosis != "v1" else Path("results")
    labels_path = folder / "failure_labels.jsonl"
    output = folder / OUT.name
    if VARIANT not in {"original", "status"}:
        raise ValueError("TOOLSCOPE_JUDGE_VARIANT must be original or status")
    rows = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
    outcomes = read_outcomes(JUDGE, HUMAN, MODEL, VARIANT)
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (labels_path, JUDGE, HUMAN) if p.exists()}
    manifest = None
    if diagnosis != "v1":
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        sources[str(manifest_path)] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    data = build_report(rows, outcomes, list(RUNS), MODEL, VARIANT, sources, diagnosis, manifest)
    output.write_text(render_markdown(data))
    # Publish the chart, membership and cases together, avoiding partial reads.
    target = output.with_suffix(".json")
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=1) + "\n")
    temporary.replace(target)
    judged = sum(1 for k in outcomes if k[:2] in RUNS)
    print(f"runs={len(RUNS)} judged={judged} labels={len(rows)} out={output}")


if __name__ == "__main__":
    main()

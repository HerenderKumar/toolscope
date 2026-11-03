"""Phase 5 Step 1: label every unsolved (run, task, capability) with D-010 precedence, plus the retrieval-only
labels for all 6 strategies. Writes results/failure_labels.jsonl (one row per label), the per-query verdicts
into logs/retrieval.labelled.jsonl (same schema as logs/retrieval.jsonl), and a count table to stdout.

Run historical v1: PYTHONPATH=src .venv/bin/python scripts/classify_failures.py
Run scoped v2 (D-078): PYTHONPATH=src .venv/bin/python scripts/classify_failures.py --version v2
The latter writes separate results/diagnosis_v2 artifacts and logs/retrieval.labelled.v2.jsonl.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from toolscope.diagnosis.classify import DESCRIPTION, LABELS, RETRIEVAL as MISS, broken_tools, label_offline, label_run
from toolscope.diagnosis.runs import RUNS, load
from toolscope.diagnosis.repair_evidence import VERSION, apply_repairs, normalize_probes

GOLD = Path("data/gold/gold.jsonl")
RETRIEVAL = Path("logs/retrieval.jsonl")
RETRIEVAL_K20 = Path("logs/retrieval_k20.jsonl")  # same rankings, deeper list (ranks 11-20 for "near miss")
OUT = Path("results/failure_labels.jsonl")
LABELLED = Path("logs/retrieval.labelled.jsonl")
WHATIF = Path("results/whatif.jsonl")  # Step 2-3 output (scripts/rewrite_probe.py)


def offline_lists(path: Path) -> dict[str, dict[str, list[str]]]:
    lists: dict[str, dict[str, list[str]]] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        lists.setdefault(r["strategy"], {})[r["task_id"]] = [t for t, _ in r["top_k"]]
    return lists


def description_faults(path: Path) -> dict[tuple[str, str], str]:
    """(task_id, cap_id) -> evidence, for capabilities whose rewritten tool moved into a top-10 (D-010 rule 5)."""
    faults: dict[tuple[str, str], str] = {}
    if not path.exists():
        return faults
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["moved_into_top10"] and (r["task_id"], r["cap_id"]) not in faults:
            faults[(r["task_id"], r["cap_id"])] = (f"whatif:{r['tool_id']} bm25 {r['bm25'][0]}->{r['bm25'][1]}, "
                                                   f"semantic {r['semantic'][0]}->{r['semantic'][1]}")
    return faults


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v1", "v2"), default="v1",
                        help="v2 writes D-078 evidence and labels to results/diagnosis_v2; v1 preserves the old pipeline")
    version = parser.parse_args().version
    gold = {json.loads(l)["task_id"]: json.loads(l) for l in GOLD.read_text().splitlines()}
    offline = offline_lists(RETRIEVAL_K20 if RETRIEVAL_K20.exists() else RETRIEVAL)
    runs = {key: load(key[0], key[1], folders) for key, folders in RUNS.items()}
    broken = broken_tools([r for recs in runs.values() for r in recs.values()])
    rows = []
    for (agent, strategy), recs in runs.items():
        for row in label_run(recs, gold, strategy, broken, offline):
            rows.append({"scope": "agent", "agent": agent, "strategy": strategy, **row})
    for row in label_offline(gold, offline):
        rows.append({"scope": "retrieval", "agent": "", **row})
    output, labelled = OUT, LABELLED
    if version == "v2":
        folder = Path("results/diagnosis_v2")
        rewrites_path, catalog_path = Path("results/rewrites.jsonl"), Path("data/catalog/tools.jsonl")
        rewrites = [json.loads(l) for l in rewrites_path.read_text().splitlines()]
        probes = [json.loads(l) for l in WHATIF.read_text().splitlines()]
        cards = {r["tool_id"]: r for r in map(json.loads, catalog_path.read_text().splitlines())}
        evidence = normalize_probes(probes, rewrites, gold, cards)
        rows = apply_repairs(rows, evidence, gold)
        prior = [json.loads(l) for l in OUT.read_text().splitlines()]
        identity = lambda r: (r["scope"], r["agent"], r["strategy"], r["task_id"], r["cap_id"])
        previous = {identity(r): r for r in prior}
        if set(previous) != {identity(r) for r in rows}:
            raise ValueError("v1 and v2 cohorts differ; review inputs before publishing")
        for r in rows:
            old = previous[identity(r)]
            if old["label"] not in {DESCRIPTION, MISS} and (old["label"], old["reason"]) != (r["label"], r["reason"]):
                raise ValueError("An earlier D-010 rule changed; review inputs before publishing")
        changes = [{"scope": r["scope"], "agent": r["agent"], "strategy": r["strategy"],
                    "task_id": r["task_id"], "cap_id": r["cap_id"],
                    "before_label": previous[identity(r)]["label"], "after_label": r["label"],
                    "before_reason": previous[identity(r)]["reason"], "after_reason": r["reason"]}
                   for r in rows if previous[identity(r)]["label"] != r["label"]]
        inputs = [GOLD, RETRIEVAL_K20 if RETRIEVAL_K20.exists() else RETRIEVAL, WHATIF, rewrites_path,
                  catalog_path, OUT, Path("src/toolscope/diagnosis/classify.py"),
                  Path("src/toolscope/diagnosis/repair_evidence.py"), Path(__file__)]
        inputs += sorted({Path(f) / "agent.jsonl" for folders in RUNS.values() for f in folders
                          if (Path(f) / "agent.jsonl").exists()})
        manifest = {"classifier_version": VERSION, "query_kind": "offline_task_question", "cutoff": 10,
                    "tested_repair_strategies": sorted({e["strategy"] for e in evidence}),
                    "agent_search_replayed": False, "probe_rows": len(probes), "evidence_rows": len(evidence),
                    "label_rows": len(rows), "changed_labels": len(changes),
                    "provenance": "Legacy probe linkage reconstructed from saved D-073 inputs; not replayed",
                    "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "repair_evidence.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in evidence))
        (folder / "changes.json").write_text(json.dumps(changes, indent=1) + "\n")
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        lines = ["# Diagnosis v2 change audit (D-078)", "", "Same frozen inputs; only description-repair evidence scope changes.",
                 "", "| Scope / run | Strategy | v1 description labels | v2 description labels |", "|---|---|---:|---:|"]
        for scope, agent, strategy in dict.fromkeys((r["scope"], r["agent"], r["strategy"]) for r in rows):
            match = lambda r: (r["scope"], r["agent"], r["strategy"]) == (scope, agent, strategy)
            before = sum(match(r) and r["label"] == DESCRIPTION for r in prior)
            after = sum(match(r) and r["label"] == DESCRIPTION for r in rows)
            lines.append(f"| {agent or scope} | {strategy} | {before} | {after} |")
        lines += ["", f"Changed labels: {len(changes)} of {len(rows)}. See changes.json for each affected capability.",
                  "Agent counts above include all tasks, not only failed tasks; the failure reports apply the unchanged outcome policy.",
                  "The four unprobed strategies are not tested for repair. Their zero description labels are not a negative experimental result.",
                  "Agent repair evidence is an offline task-question/top-10 proxy. Actual agent searches are not replayed.",
                  "Existing probes and cached descriptions are reused; query/intervention linkage is reconstructed, not newly measured."]
        (folder / "changes.md").write_text("\n".join(lines) + "\n")
        output = folder / "failure_labels.jsonl"
        labelled = Path("logs/retrieval.labelled.v2.jsonl")
    else:
        faults = description_faults(WHATIF)
        for r in rows:
            if r["label"] == MISS and r["reason"] == "pending_whatif":
                evidence = faults.get((r["task_id"], r["cap_id"]))
                r["label"], r["reason"] = (DESCRIPTION, evidence) if evidence else (MISS, "whatif_no_fix"
                                                                                if WHATIF.exists() else r["reason"])
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    # per-query verdicts for the retrieval log: {cap_id: label}
    verdicts: dict[tuple[str, str], dict[str, str]] = {}
    for r in rows:
        if r["scope"] == "retrieval":
            verdicts.setdefault((r["strategy"], r["task_id"]), {})[r["cap_id"]] = r["label"]
    with labelled.open("w") as f:
        for line in RETRIEVAL.read_text().splitlines():
            rec = json.loads(line)
            rec["verdict"] = verdicts.get((rec["strategy"], rec["task_id"]))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cols = ["solved", *LABELS]
    print("agent runs (capabilities):")
    print("  " + " | ".join(["run", *cols]))
    for agent, strategy in RUNS:
        c = Counter(r["label"] for r in rows if r["scope"] == "agent" and r["agent"] == agent
                    and r["strategy"] == strategy)
        print("  " + " | ".join([f"{agent}/{strategy}", *(str(c[x]) for x in cols)]))
    print("retrieval only (capabilities):")
    for strategy in offline:
        c = Counter(r["label"] for r in rows if r["scope"] == "retrieval" and r["strategy"] == strategy)
        print(f"  {strategy}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    caps = sum(len(t["capabilities"]) for t in gold.values())
    print(f"runs={len(runs)} capabilities={caps} rows={len(rows)} broken_tools={len(broken)} "
          f"offline_depth={len(next(iter(next(iter(offline.values())).values())))} out={output}")


if __name__ == "__main__":
    main()

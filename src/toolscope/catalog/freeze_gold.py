"""Validate the hand-reviewed answer key and freeze it.

Reads data/gold/gold_draft.jsonl + review.csv + equivalences.csv and writes:
  data/gold/gold.jsonl         per task: capabilities with `strict` and `acceptable` tool ids
  data/gold/qrels_strict.json  ranx qrels {task_id: {tool_id: 1}} from the strict tools only
  data/gold/provenance.txt     sha256 of both files + inputs, LiveMCPBench commit, timestamp
Writes nothing if any check fails. Then: git tag gold-v1.

review.csv `resolved_tool_id`: one tool_id, several space-separated (the annotation cannot
tell them apart, all count as strict), or DROP (explain in `note`).
equivalences.csv `equivalent`: yes / no for every row; yes rows merge into one class.

Run:  PYTHONPATH=src .venv/bin/python -m toolscope.catalog.freeze_gold
"""

import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from toolscope.log import get_logger

log = get_logger("toolscope.gold")

LIVEMCPBENCH_DIR = Path(os.environ.get("TOOLSCOPE_LIVEMCPBENCH_DIR", "third_party/LiveMCPBench"))
CATALOG = Path(os.environ.get("TOOLSCOPE_CATALOG", "data/catalog/tools.jsonl"))
GOLD_DIR = Path(os.environ.get("TOOLSCOPE_GOLD_DIR", "data/gold"))


def equivalence_classes(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    """Union-find over rows marked equivalent=yes. Returns tool_id -> its full class."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row in rows:
        if row["equivalent"].strip().lower() != "yes":
            continue
        ids = row["tool_ids"].split()
        for other in ids[1:]:
            parent[find(other)] = find(ids[0])

    classes: dict[str, set[str]] = {}
    for x in list(parent):
        classes.setdefault(find(x), set()).add(x)
    return {x: classes[find(x)] for x in parent}


def freeze(drafts: list[dict[str, Any]], review_rows: list[dict[str, str]],
           eq_rows: list[dict[str, str]], catalog_ids: set[str]):
    """Returns (gold records, strict qrels, problems). Problems are 'kind: detail' strings."""
    problems: list[str] = []
    for row in eq_rows:
        if row["equivalent"].strip().lower() not in ("yes", "no"):
            problems.append(f"equivalence_undecided: {row['tool_ids']}")
        for tid in row["tool_ids"].split():
            if tid not in catalog_ids:
                problems.append(f"unknown_id: {tid} (equivalences.csv)")
    classes = equivalence_classes(eq_rows)

    resolved = {r["cap_id"]: r["resolved_tool_id"].strip() for r in review_rows}
    known_caps = {c["cap_id"] for d in drafts for c in d["capabilities"]}
    problems += [f"orphan_review: {cap_id}" for cap_id in resolved if cap_id not in known_caps]

    gold, qrels = [], {}
    for d in drafts:
        caps = []
        for c in d["capabilities"]:
            if c["status"] == "auto":
                strict = [c["annotated_tool_id"]]
            else:
                value = resolved.get(c["cap_id"], "")
                if not value:
                    problems.append(f"unresolved: {c['cap_id']} {c['annotated_name']}")
                    continue
                if value.upper() == "DROP":
                    log.info("capability dropped", extra={"ctx": {"cap_id": c["cap_id"]}})
                    continue
                strict = value.split()
            bad = [t for t in strict if t not in catalog_ids]
            if bad:
                problems += [f"unknown_id: {t} ({c['cap_id']})" for t in bad]
                continue
            acceptable = set(strict)
            for t in strict:
                acceptable |= classes.get(t, set())
            caps.append({"cap_id": c["cap_id"], "step": c["step"],
                         "annotated_name": c["annotated_name"],
                         "strict": sorted(strict), "acceptable": sorted(acceptable)})
        if not caps:
            problems.append(f"empty_task: {d['task_id']}")
            continue
        gold.append({"task_id": d["task_id"], "category": d["category"],
                     "question": d["question"], "capabilities": caps})
        qrels[d["task_id"]] = {t: 1 for c in caps for t in c["strict"]}
    return gold, qrels, problems


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    drafts = [json.loads(line) for line in (GOLD_DIR / "gold_draft.jsonl").read_text().splitlines()]
    review_rows = read_csv(GOLD_DIR / "review.csv")
    eq_rows = read_csv(GOLD_DIR / "equivalences.csv")
    catalog_ids = {json.loads(line)["tool_id"] for line in CATALOG.read_text().splitlines()}

    gold, qrels, problems = freeze(drafts, review_rows, eq_rows, catalog_ids)
    counts = {k: sum(p.startswith(k + ":") for p in problems)
              for k in ("unknown_id", "empty_task", "unresolved", "equivalence_undecided", "orphan_review")}
    summary = (f"tasks={len(gold)} capabilities={sum(len(g['capabilities']) for g in gold)} "
               f"unknown_ids={counts['unknown_id']} empty_tasks={counts['empty_task']} "
               f"unresolved={counts['unresolved']} equivalence_undecided={counts['equivalence_undecided']}")
    if problems:
        for p in problems:
            log.error("gold check failed", extra={"ctx": {"problem": p}})
        print(summary + " status=NOT_FROZEN")
        sys.exit(1)

    gold_path, qrels_path = GOLD_DIR / "gold.jsonl", GOLD_DIR / "qrels_strict.json"
    with gold_path.open("w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    qrels_path.write_text(json.dumps(qrels, indent=1, sort_keys=True))
    commit = subprocess.run(["git", "-C", str(LIVEMCPBENCH_DIR), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    (GOLD_DIR / "provenance.txt").write_text("\n".join([
        f"frozen_at {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"livemcpbench_commit {commit}",
        f"sha256 {sha256(gold_path)} gold.jsonl",
        f"sha256 {sha256(qrels_path)} qrels_strict.json",
        f"sha256 {sha256(CATALOG)} {CATALOG}",
        f"sha256 {sha256(GOLD_DIR / 'review.csv')} review.csv",
        f"sha256 {sha256(GOLD_DIR / 'equivalences.csv')} equivalences.csv",
    ]) + "\n")
    print(summary + f" status=FROZEN gold_sha256={sha256(gold_path)[:12]}")


if __name__ == "__main__":
    main()

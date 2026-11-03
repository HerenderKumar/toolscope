"""Draft the answer key: one capability per annotated tool mention.

Unique exact name matches are mapped automatically. Unknown and ambiguous names go to
data/gold/review.csv for hand review. Also writes equivalences.csv, pre-filled with the tool
names shared across servers, as the starting point for adding equivalents.
Refuses to overwrite hand edits unless TOOLSCOPE_FORCE=1.

Run:  PYTHONPATH=src .venv/bin/python -m toolscope.catalog.build_gold_draft
"""

import csv
import difflib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from toolscope.log import get_logger

log = get_logger("toolscope.gold")

LIVEMCPBENCH_DIR = Path(os.environ.get("TOOLSCOPE_LIVEMCPBENCH_DIR", "third_party/LiveMCPBench"))
ANNOTATIONS = LIVEMCPBENCH_DIR / "annotated_data" / "all_annotations.json"
CATALOG = Path(os.environ.get("TOOLSCOPE_CATALOG", "data/catalog/tools.jsonl"))
GOLD_DIR = Path(os.environ.get("TOOLSCOPE_GOLD_DIR", "data/gold"))

TOOL_LINE = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$")


def parse_tool_list(text: str) -> list[str]:
    """'1. load\\n2. beat_track' -> ['load', 'beat_track']. Non-numbered lines are logged and skipped."""
    names = []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = TOOL_LINE.match(line)
        if m:
            names.append(m.group(1))
        else:
            log.warning("unparsed tool line", extra={"ctx": {"line": line}})
    return names


def index_by_name(cards: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_name: dict[str, list[str]] = defaultdict(list)
    for c in cards:
        by_name[c["tool_name"]].append(c["tool_id"])
    return by_name


def suggest(name: str, candidates: list[str], task_servers: set[str],
            cards: list[dict[str, Any]]) -> list[str]:
    """Hints for the reviewer, never applied automatically.

    Ambiguous: candidates whose server another tool in the same task uses come first.
    Unknown: close tool-name matches, then tools on close-matching servers.
    """
    if candidates:
        return sorted(candidates, key=lambda t: t.split("/")[0] not in task_servers)
    names = sorted({c["tool_name"] for c in cards})
    keys = sorted({c["server_key"] for c in cards})
    out = [c["tool_id"] for n in difflib.get_close_matches(name, names, 4, 0.5)
           for c in cards if c["tool_name"] == n]
    for key in difflib.get_close_matches(name, keys, 2, 0.5):
        out += [c["tool_id"] for c in cards if c["server_key"] == key]
    return list(dict.fromkeys(out))[:6]


def build(tasks: list[dict[str, Any]], cards: list[dict[str, Any]]):
    by_name = index_by_name(cards)
    drafts, review = [], []
    for task in tasks:
        names = parse_tool_list(task["Annotator Metadata"]["Tools"])
        unique_servers = {by_name[n][0].split("/")[0] for n in names if len(by_name.get(n, [])) == 1}
        capabilities = []
        for i, name in enumerate(names, start=1):
            matches = by_name.get(name, [])
            cap_id = f"{task['task_id']}#{i}"
            status = "auto" if len(matches) == 1 else ("ambiguous" if matches else "unknown")
            capabilities.append({
                "cap_id": cap_id,
                "step": i,
                "annotated_name": name,
                "annotated_tool_id": matches[0] if status == "auto" else None,
                "status": status,
            })
            if status != "auto":
                review.append({
                    "task_id": task["task_id"],
                    "cap_id": cap_id,
                    "annotated_name": name,
                    "issue": status,
                    "candidates": " ".join(matches),
                    "suggested": " ".join(suggest(name, matches, unique_servers, cards)),
                    "question": task["Question"].replace("\n", " ")[:300],
                    "resolved_tool_id": "",
                    "note": "",
                })
        if str(len(names)) != str(task["Annotator Metadata"]["Number of tools"]).strip():
            log.warning("Number of tools differs from Tools list; using the list",
                        extra={"ctx": {"task_id": task["task_id"], "listed": len(names),
                                       "number_of_tools": task["Annotator Metadata"]["Number of tools"]}})
        drafts.append({
            "task_id": task["task_id"],
            "category": task["category"],
            "question": task["Question"],
            "capabilities": capabilities,
        })
    shared = {n: ids for n, ids in by_name.items() if len(ids) > 1}
    return drafts, review, shared


def has_hand_edits(path: Path, column: str) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as f:
        return any((row.get(column) or "").strip() for row in csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    tasks = json.loads(ANNOTATIONS.read_text())
    cards = [json.loads(line) for line in CATALOG.read_text().splitlines()]
    drafts, review, shared = build(tasks, cards)

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    edited = [p for p, col in ((GOLD_DIR / "review.csv", "resolved_tool_id"),
                               (GOLD_DIR / "equivalences.csv", "equivalent"))
              if has_hand_edits(p, col)]
    if edited and os.environ.get("TOOLSCOPE_FORCE") != "1":
        raise SystemExit(f"refusing to overwrite hand edits in {[str(p) for p in edited]}; "
                         "set TOOLSCOPE_FORCE=1 to start over")
    with (GOLD_DIR / "gold_draft.jsonl").open("w", encoding="utf-8") as f:
        for d in drafts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    write_csv(GOLD_DIR / "review.csv", review,
              ["task_id", "cap_id", "annotated_name", "issue", "candidates", "suggested",
               "question", "resolved_tool_id", "note"])
    write_csv(GOLD_DIR / "equivalences.csv",
              [{"tool_name": n, "tool_ids": " ".join(ids), "equivalent": "", "rationale": ""}
               for n, ids in sorted(shared.items())],
              ["tool_name", "tool_ids", "equivalent", "rationale"])

    caps = [c for d in drafts for c in d["capabilities"]]
    n_auto = sum(c["status"] == "auto" for c in caps)
    n_unknown = sum(c["status"] == "unknown" for c in caps)
    n_amb = sum(c["status"] == "ambiguous" for c in caps)
    print(f"tasks={len(drafts)} mentions={len(caps)} auto_mapped={n_auto} "
          f"({n_auto / len(caps):.0%}) ambiguous={n_amb} unknown={n_unknown} "
          f"review_rows={len(review)} shared_names={len(shared)}")


if __name__ == "__main__":
    main()

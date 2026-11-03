"""Merge the author's task-success labels into one CSV (D-072): the first-pass sheet, overridden by the
re-label sheet for the tasks it holds. Prints success counts on all 95 tasks and on the 81 solvable ones (D-057).

Run:  PYTHONPATH=src .venv/bin/python scripts/merge_human_labels.py FIRST.xlsx RELABEL.xlsx OUT.csv AGENT STRATEGY
e.g.  ... data/human/label_sheet_kimi_bm25_reviewed.xlsx data/human/relabel_32_kimi_bm25_reviewed.xlsx \
          data/human/human_labels.csv "Kimi K2.6" bm25
"""

import csv
import sys
from pathlib import Path

from openpyxl import load_workbook

# D-057: every acceptable tool of one capability is broken upstream
UNSOLVABLE = {"0180954e", "00e13e2e", "0e3287cb", "1a3ffa81", "41899b5b", "470f7ac1", "4cbd9682", "4d7582c4",
              "5b2087e4", "629106a5", "a984d31b", "ef7126b3", "f2a9eb67", "f2b074b5"}
FIELDS = ["task_id", "agent", "strategy", "label", "source", "note"]


def read(path: Path) -> dict[str, dict]:
    ws = load_workbook(path)["labels"]
    header = [c.value for c in ws[1]]
    rows = [dict(zip(header, r)) for r in ws.iter_rows(min_row=2, values_only=True)]
    return {r["task_id"]: r for r in rows}


def main() -> None:
    first, relabel, out, agent, strategy = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), *sys.argv[4:6]
    merged = {}
    for source, path in (("first_pass", first), ("relabel", relabel)):
        for tid, r in read(path).items():
            label = str(r["label"] or "").strip().lower()
            if label not in ("success", "failure"):
                raise SystemExit(f"{path.name}: task {tid} has no valid label ({r['label']!r})")
            merged[tid] = {"task_id": tid, "agent": agent, "strategy": strategy, "label": label,
                           "source": source, "note": r.get("note") or ""}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(merged[t] for t in sorted(merged))
    ok = [t for t, r in merged.items() if r["label"] == "success"]
    solvable = [t for t in merged if t[:8] not in UNSOLVABLE]
    ok_solvable = [t for t in ok if t[:8] not in UNSOLVABLE]
    print(f"tasks={len(merged)} success={len(ok)} ({len(ok) / len(merged):.1%}) "
          f"solvable={len(solvable)} success_solvable={len(ok_solvable)} ({len(ok_solvable) / len(solvable):.1%}) "
          f"relabelled={sum(r['source'] == 'relabel' for r in merged.values())} out={out}")


if __name__ == "__main__":
    main()

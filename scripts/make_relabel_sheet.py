"""Write a re-label sheet for chosen tasks of a reviewed labelling sheet (D-072): the same columns, the full
logged result snippet (500 chars, the log's own limit) instead of 300, the LiveMCPEval rule on top, and the
author's previous label/note kept for reference.

Run:  PYTHONPATH=src .venv/bin/python scripts/make_relabel_sheet.py REVIEWED.xlsx OUT.xlsx IDS LOG [LOG ...]
IDS is a comma-separated list of task-id prefixes.
"""

import json
import sys
from pathlib import Path

from openpyxl import load_workbook

from toolscope.eval.label_sheet import INSTRUCTIONS, build_rows, latest_records, write_xlsx

ANNOTATIONS = Path("third_party/LiveMCPBench/annotated_data/all_annotations.json")
LOG_SNIPPET = 500  # the agent log keeps at most 500 characters of each tool result
RULE = [
    "RE-LABEL RULE (LiveMCPEval's judge sees the tool calls, tool descriptions and final answer, not the results):",
    "success = for every key point the agent called a suitable tool and the call succeeded, and the final answer "
    "completes the task. A fact you cannot see in the 500-char excerpt is NOT by itself a failure.",
    "failure = a key point was clearly not done (no suitable call, call failed and not recovered, file not "
    "written), or the agent says it used its own knowledge instead of tools.",
]


def main() -> None:
    reviewed, out, ids = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",")
    logs = [Path(p) for p in sys.argv[4:]]
    ws = load_workbook(reviewed)["labels"]
    header = [c.value for c in ws[1]]
    previous = {r[header.index("task_id")]: (r[header.index("label")], r[header.index("note")])
                for r in ws.iter_rows(min_row=2, values_only=True)}
    records = {t: r for t, r in latest_records(logs).items() if r.get("strategy") == "bm25"}
    tasks = [t for t in json.loads(ANNOTATIONS.read_text()) if any(t["task_id"].startswith(i) for i in ids)]
    rows = build_rows(records, tasks, snippet=LOG_SNIPPET)
    for row in rows:
        old_label, old_note = previous.get(row["task_id"], ("", ""))
        row["previous_label"], row["previous_note"] = old_label, old_note
    INSTRUCTIONS[:0] = RULE
    write_xlsx(rows, out, f"Re-label {len(rows)} tasks (Kimi bm25)")
    print(f"tasks={len(rows)} requested={len(ids)} out={out}")


if __name__ == "__main__":
    main()

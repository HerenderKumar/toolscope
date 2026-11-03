"""Write the task-success labelling sheet for one agent run (D-072).

Run:  PYTHONPATH=src .venv/bin/python scripts/make_label_sheet.py OUT.xlsx LOG [LOG ...]
e.g.  PYTHONPATH=src .venv/bin/python scripts/make_label_sheet.py data/human/label_sheet_kimi_bm25.xlsx \
          logs/kimi/test5/agent.jsonl logs/kimi/d063_bm25_a/agent.jsonl logs/kimi/d063_bm25_b/agent.jsonl
Only bm25 records are used (test5 also holds bm25 only). The latest record per task wins.
"""

import json
import sys
from pathlib import Path

from toolscope.eval.label_sheet import build_rows, latest_records, write_xlsx

ANNOTATIONS = Path("third_party/LiveMCPBench/annotated_data/all_annotations.json")


def main() -> None:
    out, logs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    records = {t: r for t, r in latest_records(logs).items() if r.get("strategy") == "bm25"}
    tasks = json.loads(ANNOTATIONS.read_text())
    rows = build_rows(records, tasks)
    model = next(iter(records.values())).get("model", "?") if records else "?"
    write_xlsx(rows, out, f"Task-success labels: {model}, bm25 ({len(rows)} tasks)")
    missing = sum(r["stop_reason"] == "missing" for r in rows)
    print(f"tasks={len(rows)} with_record={len(rows) - missing} missing={missing} out={out}")


if __name__ == "__main__":
    main()

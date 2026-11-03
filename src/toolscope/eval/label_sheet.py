"""Build the human task-success labelling sheet (D-072): one row per task of one agent run, showing what a
LiveMCPEval judge sees (task, key points, tool calls with results, final answer), and an empty label column.
No verdicts from Claude, the judge or paid_run_log.md are shown, so the labels stay independent."""

import json
from pathlib import Path
from typing import Any

LABELS = ("success", "failure")
MAX_CELL = 32000  # Excel's cell limit is 32,767 characters
SNIPPET = 300

# LiveMCPEval's own criteria (third_party/LiveMCPBench/evaluator/llm_as_judge_baseline.py, livemcp_eval prompt)
INSTRUCTIONS = [
    "Label each task 'success' or 'failure' in the label column (dropdown). Add a note when unsure.",
    "Use the same rules as LiveMCPEval, the judge prompt of LiveMCPBench:",
    "1. Success only if the agent completed the task and met all requirements (check every key point).",
    "2. Information must come from tool calls. If the agent got it from its own internal knowledge, "
    "it counts as failed.",
    "3. Some tasks require creating files to count as successful (look for a successful write/create call).",
    "Key points = the annotators' steps (LiveMCPBench's default key points). The expected answer is the "
    "annotators' reference.",
    "Label before looking at any judge output. Do not edit the other columns.",
]


def latest_records(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Latest record per task across agent logs (by ts), so a re-run overrides an earlier record."""
    best: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            old = best.get(rec["task_id"])
            if old is None or rec.get("ts", "") >= old.get("ts", ""):
                best[rec["task_id"]] = rec
    return best


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def format_calls(steps: list[dict[str, Any]], snippet: int = SNIPPET) -> str:
    """Numbered tool calls: tool_id(args) -> OK: result snippet | ERROR: message. Searches are listed briefly."""
    lines = []
    for step in steps:
        if step.get("type") == "search":
            lines.append(f"[search] {_clip(step.get('query', ''), 120)}")
        elif step.get("type") == "call":
            args = _clip(json.dumps(step.get("args", {}), ensure_ascii=False), 200)
            if step.get("ok"):
                outcome = "OK: " + _clip(step.get("result_snippet") or "", snippet)
            else:
                outcome = "ERROR: " + _clip(step.get("error") or "", snippet)
            lines.append(f"{len([x for x in lines if not x.startswith('[search]')]) + 1}. "
                         f"{step.get('tool_id')}({args}) -> {outcome}")
    return "\n".join(lines) if lines else "(no tool calls)"


def build_rows(records: dict[str, dict[str, Any]], tasks: list[dict[str, Any]],
               snippet: int = SNIPPET) -> list[dict[str, Any]]:
    """One row per annotated task, in annotation order; tasks without a record are marked missing."""
    rows = []
    for i, task in enumerate(tasks, 1):
        rec = records.get(task["task_id"])
        meta = task.get("Annotator Metadata", {})
        rows.append({
            "n": i,
            "task_id": task["task_id"],
            "category": task.get("category", ""),
            "question": task["Question"][:MAX_CELL],
            "key_points": str(meta.get("Steps", ""))[:MAX_CELL],
            "expected_answer": str(task.get("answers", ""))[:MAX_CELL],
            "tool_calls": format_calls(rec["steps"], snippet)[:MAX_CELL] if rec else "(no record)",
            "final_answer": str(rec.get("final_answer") or "(empty)")[:MAX_CELL] if rec else "(no record)",
            "stop_reason": rec.get("stop_reason", "") if rec else "missing",
            "label": "",
            "note": "",
        })
    return rows


def write_xlsx(rows: list[dict[str, Any]], out: Path, title: str) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "labels"
    cols = list(rows[0].keys())
    widths = {"n": 5, "task_id": 12, "category": 12, "question": 45, "key_points": 45, "expected_answer": 25,
              "tool_calls": 70, "final_answer": 70, "stop_reason": 11, "label": 11, "note": 30}
    ws.append(cols)
    for c, name in enumerate(cols, 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="FFF2CC" if name in ("label", "note") else "DDEBF7")
        ws.column_dimensions[cell.column_letter].width = widths.get(name, 15)
    for row in rows:
        ws.append([row[c] for c in cols])
    wrap = Alignment(wrap_text=True, vertical="top")
    for r in ws.iter_rows(min_row=2):
        for cell in r:
            cell.alignment = wrap
        ws.row_dimensions[r[0].row].height = 150
    label_col = ws.cell(row=1, column=cols.index("label") + 1).column_letter
    dv = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{label_col}2:{label_col}{len(rows) + 1}")
    ws.freeze_panes = "E2"
    info = wb.create_sheet("instructions", 0)
    info.column_dimensions["A"].width = 120
    info.append([title])
    info["A1"].font = Font(bold=True, size=13)
    for line in INSTRUCTIONS:
        info.append([line])
    for r in info.iter_rows():
        r[0].alignment = Alignment(wrap_text=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)

import json

from openpyxl import load_workbook

from toolscope.eval.label_sheet import build_rows, format_calls, latest_records, write_xlsx


def _rec(task_id, ts, answer):
    return {"task_id": task_id, "ts": ts, "strategy": "bm25", "final_answer": answer, "stop_reason": "final",
            "steps": [{"type": "search", "query": "q"},
                      {"type": "call", "tool_id": "s/t", "args": {"a": 1}, "ok": True, "result_snippet": "r"},
                      {"type": "call", "tool_id": "s/u", "args": {}, "ok": False, "error": "boom"}]}


def test_latest_record_wins(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps(_rec("t1", "2026-11-01T01", "old")) + "\n")
    b.write_text(json.dumps(_rec("t1", "2026-11-01T02", "new")) + "\n")
    assert latest_records([b, a])["t1"]["final_answer"] == "new"


def test_format_calls_numbers_calls_not_searches():
    text = format_calls(_rec("t1", "x", "y")["steps"])
    assert text.splitlines() == ["[search] q", '1. s/t({"a": 1}) -> OK: r', "2. s/u({}) -> ERROR: boom"]


def test_rows_and_xlsx(tmp_path):
    tasks = [{"task_id": "t1", "Question": "Q1", "answers": "A", "category": "C",
              "Annotator Metadata": {"Steps": "1. do"}},
             {"task_id": "t2", "Question": "Q2", "answers": "", "category": "C", "Annotator Metadata": {}}]
    rows = build_rows({"t1": _rec("t1", "x", "done")}, tasks)
    assert rows[0]["final_answer"] == "done" and rows[0]["label"] == ""
    assert rows[1]["stop_reason"] == "missing"
    out = tmp_path / "s.xlsx"
    write_xlsx(rows, out, "title")
    wb = load_workbook(out)
    assert wb.sheetnames == ["instructions", "labels"]
    assert wb["labels"].cell(row=2, column=2).value == "t1"

"""Phase 5 Step 7: agreement between the LiveMCPEval judge (logs/judge.jsonl) and the human task-success labels.
Human sources: data/human/human_labels.csv (Kimi bm25, 95 tasks) and, when labelled, the blind 60-task sample
(its reviewed .xlsx + data/human/sample60_key.json). Writes results/judge_agreement.md.

Run:  PYTHONPATH=src .venv/bin/python scripts/kappa.py [SAMPLE_REVIEWED.xlsx]
"""

import csv
import json
import os
import sys
from pathlib import Path

from openpyxl import load_workbook

from toolscope.judge.agreement import KAPPA_MIN, agreement_report

MODEL = os.environ.get("TOOLSCOPE_JUDGE_MODEL", "deepseek/deepseek-v3.2")
VARIANT = os.environ.get("TOOLSCOPE_JUDGE_VARIANT", "original")  # or "status" (D-074)
JUDGE = Path(os.environ.get("TOOLSCOPE_JUDGE_LOG", "logs/judge.jsonl"))
HUMAN = Path("data/human/human_labels.csv")
SAMPLE_KEY = Path("data/human/sample60_key.json")
OUT = Path("results/judge_agreement.md" if VARIANT == "original" else f"results/judge_agreement_{VARIANT}.md")


def judge_verdicts() -> dict[tuple[str, str, str], str]:
    out = {}
    for line in JUDGE.read_text().splitlines():
        r = json.loads(line)
        if r["model"] == MODEL and r.get("variant", "original") == VARIANT and r["verdict"] in ("success", "failure", "unparsed"):
            out[(r["agent"], r["strategy"], r["task_id"])] = "success" if r["verdict"] == "success" else "failure"
    return out


def human_labels(sample: Path | None) -> list[tuple[tuple[str, str, str], str, str]]:
    """[(key, label, source)] from the CSV and, optionally, the labelled blind sample."""
    rows = [((r["agent"], r["strategy"], r["task_id"]), r["label"], "kimi_bm25_full")
            for r in csv.DictReader(HUMAN.open())]
    if sample and sample.exists():
        key = json.loads(SAMPLE_KEY.read_text())
        ws = load_workbook(sample)["labels"]
        header = [c.value for c in ws[1]]
        for r in ws.iter_rows(min_row=2, values_only=True):
            r = dict(zip(header, r))
            label = str(r["label"] or "").strip().lower()
            if label in ("success", "failure"):
                k = key[r["n"]]
                rows.append(((k["agent"], k["strategy"], k["task_id"]), label, "sample60"))
    return rows


def main() -> None:
    sample = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    verdicts = judge_verdicts()
    labels = human_labels(sample)
    groups = {"all": labels, **{s: [x for x in labels if x[2] == s] for s in sorted({x[2] for x in labels})}}
    lines = [f"## Judge agreement with human task-success labels (judge {MODEL}, variant {VARIANT})", "",
             f"Trust rule (D-072): the judge is used only if Cohen's kappa >= {KAPPA_MIN}. LiveMCPBench reports raw "
             "agreement (DeepSeek-V3-0324: 81.05%).", "",
             "| Labels | n | Raw agreement | Cohen's kappa (95% CI) | Passes | Human success | Judge success | Confusion H/J: f-f, f-s, s-f, s-s |",
             "|---|---|---|---|---|---|---|---|"]
    missing = 0
    data = []
    for name, rows in groups.items():
        pairs = [(lab, verdicts[k]) for k, lab, _ in rows if k in verdicts]
        missing += (len(rows) - len(pairs)) if name == "all" else 0
        if not pairs:
            continue
        rep = agreement_report(pairs)
        lo, hi = rep["kappa_ci"]
        cm = rep["confusion"]
        cells = ", ".join(str(cm.get(f"human={h}/judge={j}", 0)) for h, j in
                          (("failure", "failure"), ("failure", "success"), ("success", "failure"), ("success", "success")))
        lines.append(f"| {name} | {rep['n']} | {rep['agreement']:.1%} | {rep['kappa']:.3f} ({lo:.2f}–{hi:.2f}) | "
                     f"{'yes' if rep['passes'] else 'no'} | {sum(h == 'success' for h, _ in pairs)} "
                     f"| {sum(j == 'success' for _, j in pairs)} | {cells} |")
        data.append({"labels": name, **rep, "human_success": sum(h == "success" for h, _ in pairs),
                     "judge_success": sum(j == "success" for _, j in pairs)})
        if name == "all":
            overall = rep
    lines += ["", "Confusion (all): " + ", ".join(f"{k}: {v}" for k, v in sorted(overall["confusion"].items()))]
    OUT.write_text("\n".join(lines) + "\n")
    OUT.with_suffix(".json").write_text(json.dumps({"variant": VARIANT, "groups": data}, indent=1) + "\n")
    print(f"pairs={overall['n']} missing_verdicts={missing} agreement={overall['agreement']:.3f} "
          f"kappa={overall['kappa']:.3f} passes={overall['passes']} out={OUT}")


if __name__ == "__main__":
    main()

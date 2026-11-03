"""Evaluate the retrieval logs (no re-retrieval) and write results/table.md, table.tex, metrics.json, per_query.jsonl.

Also evaluates the D-017 ablation log (ToolRet with the train.sh configuration) if present.
Run:  PYTHONPATH=src .venv/bin/python scripts/evaluate.py
"""

import json
import os
from pathlib import Path

from toolscope.eval.report import ALL_METRICS, LABELS, evaluate_logs, render_latex, render_markdown
from toolscope.retrievers.base import STRATEGIES

LOG = Path(os.environ.get("TOOLSCOPE_RETRIEVAL_LOG", "logs/retrieval.jsonl"))
ABLATION_LOG = Path(os.environ.get("TOOLSCOPE_ABLATION_LOG", "logs/ablation_toolret_cls.jsonl"))
GOLD = Path(os.environ.get("TOOLSCOPE_GOLD", "data/gold/gold.jsonl"))
QRELS = Path(os.environ.get("TOOLSCOPE_QRELS", "data/gold/qrels_strict.json"))
OUT = Path(os.environ.get("TOOLSCOPE_RESULTS_DIR", "results"))
BASELINE = "bm25"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def ablation_section(gold, qrels) -> str:
    if not ABLATION_LOG.exists():
        return ""
    res = evaluate_logs(read_jsonl(ABLATION_LOG), gold, qrels, ["toolret"], baseline="toolret")
    cells = " | ".join(f"{res['toolret']['mean'][m]:.3f}" for m in ALL_METRICS)
    return "\n".join([
        "## Ablation: ToolRet-trained-e5 configuration (D-017)", "",
        "Main table: configuration of ToolRet's evaluation code (mean pooling, `Instruct: \\nquery: `). "
        "Row below: configuration of ToolRet's training script (CLS pooling, `Represent this sentence…`), "
        f"from `{ABLATION_LOG}`.", "",
        "| Configuration | " + " | ".join(LABELS[m] for m in ALL_METRICS) + " |",
        "|" + "---|" * (len(ALL_METRICS) + 1),
        f"| toolret (train.sh: CLS) | {cells} |", ""])


def main() -> None:
    lines = read_jsonl(LOG)
    gold = read_jsonl(GOLD)
    qrels = json.loads(QRELS.read_text())
    strategies = [s for s in STRATEGIES if any(r["strategy"] == s for r in lines)]
    res = evaluate_logs(lines, gold, qrels, strategies, BASELINE)
    n_cmp = len(strategies) - 1

    OUT.mkdir(parents=True, exist_ok=True)
    md = render_markdown(res, strategies, len(qrels), n_cmp, BASELINE) + "\n" + ablation_section(gold, qrels)
    (OUT / "table.md").write_text(md + f"\nSource: `{LOG}` (run_id {lines[0]['run_id']}).\n")
    (OUT / "table.tex").write_text(render_latex(res, strategies, len(qrels), BASELINE))
    (OUT / "metrics.json").write_text(json.dumps(  # the same numbers as table.md, for the read-only API/UI
        {"baseline": BASELINE, "n_queries": len(qrels), "n_comparisons": n_cmp,
         "strategies": {s: {k: res[s][k] for k in ("mean", "delta", "p_raw", "p_bonf")} for s in strategies}},
        indent=1) + "\n")
    task_ids = sorted(qrels)
    with (OUT / "per_query.jsonl").open("w") as f:
        for s in strategies:
            for i, t in enumerate(task_ids):
                f.write(json.dumps({"strategy": s, "task_id": t,
                                    **{m: res[s]["per_query"][m][i] for m in ALL_METRICS}}) + "\n")
    sig = sum(res[s]["p_bonf"].get(m, 1) < 0.05 for s in strategies for m in ALL_METRICS)
    best = max(strategies, key=lambda s: res[s]["mean"]["ndcg@10"])
    print(f"strategies={len(strategies)} queries={len(qrels)} metrics={len(ALL_METRICS)} "
          f"significant_cells={sig} best_ndcg10={best}:{res[best]['mean']['ndcg@10']:.3f} out={OUT}")


if __name__ == "__main__":
    main()

"""Write a blind task-success labelling sheet for a random sample of other agent runs (D-072 extension):
60 distinct tasks, stratified by LiveMCPBench category, spread over five runs (Sonnet bm25 20, Qwen3-235B
bm25/semantic 10+10, 7B bm25/semantic 10+10). Rows are shuffled and the agent is hidden; the key
(sample_id -> run, task) is written next to the sheet. Seeded, so the sample is reproducible.

Run:  PYTHONPATH=src .venv/bin/python scripts/make_sample_sheet.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from toolscope.diagnosis.runs import RUNS, load
from toolscope.eval.label_sheet import INSTRUCTIONS, build_rows, write_xlsx

SEED = 2027
N = 60
QUOTA = {("Sonnet 5.5", "bm25"): 20, ("Qwen3-235B", "bm25"): 10, ("Qwen3-235B", "semantic"): 10,
         ("Qwen2.5-7B (local)", "bm25"): 10, ("Qwen2.5-7B (local)", "semantic"): 10}
ANNOTATIONS = Path("third_party/LiveMCPBench/annotated_data/all_annotations.json")
OUT = Path("data/human/sample60_other_agents.xlsx")
KEY = Path("data/human/sample60_key.json")
RULE = [
    "RULE (LiveMCPEval: the judge sees the tool calls, tool descriptions and final answer, not the results):",
    "success = for every key point the agent called a suitable tool and the call succeeded, and the final answer "
    "completes the task. A fact you cannot see in the 500-char excerpt is NOT by itself a failure.",
    "failure = a key point was clearly not done (no suitable call, call failed and not recovered, file not "
    "written), or the agent says it used its own knowledge instead of tools.",
    "The agent is hidden on purpose (blind labelling). Rows are in random order.",
]


def stratified_tasks(tasks: list[dict], n: int, rng: random.Random) -> list[dict]:
    """n distinct tasks, per-category counts proportional to the category sizes (largest remainder)."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        by_cat[t["category"]].append(t)
    exact = {c: n * len(ts) / len(tasks) for c, ts in by_cat.items()}
    counts = {c: int(v) for c, v in exact.items()}
    for c in sorted(exact, key=lambda c: exact[c] - counts[c], reverse=True)[: n - sum(counts.values())]:
        counts[c] += 1
    chosen = []
    for c in sorted(by_cat):
        chosen += rng.sample(by_cat[c], counts[c])
    return chosen


def main() -> None:
    rng = random.Random(SEED)
    tasks = json.loads(ANNOTATIONS.read_text())
    chosen = stratified_tasks(tasks, N, rng)
    rng.shuffle(chosen)
    slots = [run for run, q in QUOTA.items() for _ in range(q)]
    records = {run: load(run[0], run[1], RUNS[run]) for run in QUOTA}
    rows, key = [], {}
    for i, (task, run) in enumerate(zip(chosen, slots), 1):
        row = build_rows(records[run], [task], snippet=500)[0]
        sample_id = f"S{i:02d}"
        row["n"] = sample_id
        key[sample_id] = {"agent": run[0], "strategy": run[1], "task_id": task["task_id"]}
        rows.append(row)
    order = list(range(len(rows)))
    rng.shuffle(order)
    rows = [rows[i] for i in order]
    INSTRUCTIONS[:0] = RULE
    write_xlsx(rows, OUT, f"Blind task-success labels: {len(rows)} sampled runs of other agents")
    KEY.write_text(json.dumps(key, indent=1) + "\n")
    cats = defaultdict(int)
    for t in chosen:
        cats[t["category"]] += 1
    missing = sum(r["stop_reason"] == "missing" for r in rows)
    print(f"rows={len(rows)} distinct_tasks={len({k['task_id'] for k in key.values()})} missing={missing} "
          f"categories={dict(sorted(cats.items()))} out={OUT}")


if __name__ == "__main__":
    main()

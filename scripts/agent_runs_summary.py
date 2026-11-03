"""Summarise the final D-063 agent runs: latest record per (run, task) across each run's folders, then per run
the stop reasons, tool use, gold-tool use (gold-v1 equivalence classes) and token cost. Writes
results/agent_runs.md. Costs cover the final records only (earlier attempts are not counted); the aicredits
dashboard is authoritative.

Run:  PYTHONPATH=src .venv/bin/python scripts/agent_runs_summary.py
"""

import json
from pathlib import Path

from toolscope.diagnosis.runs import RUNS, load

# ₹ per million tokens (in, out): Kimi and Sonnet = billed rates measured 31 Oct; Qwen = listed $0.46/$1.82
# at ≈ ₹118/$ (estimate); the local 7B is free.
RATES = {"Kimi K2.6": (72.4, 428.8), "Sonnet 5.5": (226.0, 1130.0), "Qwen3-235B": (54.3, 214.8),
         "Qwen2.5-7B (local)": (0.0, 0.0)}


def summarise(recs: dict[str, dict], gold: dict[str, dict], rates: tuple[float, float]) -> dict:
    s = {"n": len(recs), "final": 0, "cap": 0, "other": 0, "no_tool": 0, "with_err": 0, "calls": 0,
         "errs": 0, "gold_any": 0, "gold_all": 0, "tin": 0, "tout": 0}
    for tid, r in recs.items():
        stop = r.get("stop_reason")
        s["final" if stop == "final" else "cap" if stop == "max_steps" else "other"] += 1
        calls = [x for x in r["steps"] if x.get("type") == "call"]
        ok_tools = {x["tool_id"] for x in calls if x.get("ok")}
        s["no_tool"] += not calls
        s["with_err"] += any(not x.get("ok") for x in calls)
        s["calls"] += len(calls)
        s["errs"] += sum(not x.get("ok") for x in calls)
        caps = gold[tid]["capabilities"]
        hit = [bool(ok_tools & set(c["acceptable"])) for c in caps]
        s["gold_any"] += any(hit)
        s["gold_all"] += all(hit)
        s["tin"] += r.get("input_tokens", 0)
        s["tout"] += r.get("output_tokens", 0)
    s["cost"] = s["tin"] / 1e6 * rates[0] + s["tout"] / 1e6 * rates[1]
    return s


def main() -> None:
    gold = {json.loads(l)["task_id"]: json.loads(l) for l in Path("data/gold/gold.jsonl").read_text().splitlines()}
    lines = ["## Agent runs (D-063 settings, final record per task)", "",
             "Gold tool used = at least one *successful* call to an acceptable tool (gold-v1 classes); all "
             "capabilities = every capability of the task got a successful acceptable call. These are tool-use "
             "measures, not task success (LiveMCPEval comes in Phase 5). Cost covers final records only.", "",
             "| Agent | Strategy | Tasks | Final | Step cap | No tool call | Tasks with a tool error | Calls "
             "| Call error rate | Gold tool used | All capabilities | Tokens in / out | Cost ₹ |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    total = 0.0
    table = []
    for (run, strategy), folders in RUNS.items():
        s = summarise(load(run, strategy, folders), gold, RATES[run])
        table.append({"agent": run, "strategy": strategy, **s})
        total += s["cost"]
        rate = s["errs"] / s["calls"] if s["calls"] else 0
        lines.append(f"| {run} | {strategy} | {s['n']} | {s['final']} | {s['cap']} | {s['no_tool']} | {s['with_err']} "
                     f"| {s['calls']} | {rate:.0%} | {s['gold_any']} | {s['gold_all']} "
                     f"| {s['tin'] / 1e6:.2f}M / {s['tout'] / 1e6:.2f}M | {s['cost']:,.0f} |")
    lines += ["", f"Final-record cost ≈ ₹{total:,.0f} (Qwen at listed prices; the aicredits dashboard is "
              "authoritative and also includes earlier, superseded attempts)."]
    Path("results/agent_runs.md").write_text("\n".join(lines) + "\n")
    Path("results/agent_runs.json").write_text(json.dumps(table, indent=1) + "\n")
    print(f"runs={len(RUNS)} cost_inr={total:.0f} out=results/agent_runs.md")


if __name__ == "__main__":
    main()

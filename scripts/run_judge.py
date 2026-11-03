"""Phase 5 Step 6: score task success of every final agent record with LiveMCPEval on DeepSeek V3.2 (D-071/D-072).
Appends one line per trajectory to logs/judge.jsonl (resumable: judged (agent, strategy, task) are skipped) and
writes results/task_success.md (success on 95 tasks and on the 81 solvable ones, D-057).

Run:  PYTHONPATH=src .venv/bin/python scripts/run_judge.py [--limit N] [--workers 6] [--variant status]
--variant status = D-074 (call status added to each tool-call line); its summary goes to
results/task_success_status.md.
Temperature 0 (LiveMCPBench's ChatModel default is 0.7; 0 keeps the verdicts reproducible).
"""

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from toolscope.agent.backends import openai_chat
from toolscope.diagnosis.runs import RUNS, load
from toolscope.gateway.server import load_env_file
from toolscope.judge.livemcpeval import judge_messages, parse_verdict, tool_map
from toolscope.log import get_logger

log = get_logger("toolscope.run_judge")
MODEL = os.environ.get("TOOLSCOPE_JUDGE_MODEL", "deepseek/deepseek-v3.2")
TEMPERATURE = float(os.environ.get("TOOLSCOPE_JUDGE_TEMPERATURE", "0"))
ANNOTATIONS = Path("third_party/LiveMCPBench/annotated_data/all_annotations.json")
OUT = Path(os.environ.get("TOOLSCOPE_JUDGE_LOG", "logs/judge.jsonl"))
SUMMARY = {"original": Path("results/task_success.md"), "status": Path("results/task_success_status.md")}
UNSOLVABLE = {"0180954e", "00e13e2e", "0e3287cb", "1a3ffa81", "41899b5b", "470f7ac1", "4cbd9682", "4d7582c4",
              "5b2087e4", "629106a5", "a984d31b", "ef7126b3", "f2a9eb67", "f2b074b5"}  # D-057
# ₹ per million tokens, listed $0.27 / $0.40 (D-037) at ≈ ₹118/$; the aicredits dashboard is authoritative
RATE_IN, RATE_OUT = 31.9, 47.2


def judged(variant: str) -> dict[tuple[str, str, str], dict]:
    done = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            if r["model"] == MODEL and r.get("variant", "original") == variant and r["verdict"] != "error":
                done[(r["agent"], r["strategy"], r["task_id"])] = r
    return done


def summarise(done: dict[tuple[str, str, str], dict], variant: str) -> str:
    title = "LiveMCPEval prompt" if variant == "original" else "LiveMCPEval prompt + call status (D-074)"
    lines = [f"## Task success ({title}, judge {MODEL})", "",
             "| Agent | Strategy | Judged | Success /95 | Success /81 solvable |", "|---|---|---|---|---|"]
    for agent, strategy in RUNS:
        rows = [r for (a, s, _), r in done.items() if a == agent and s == strategy]
        ok = [r for r in rows if r["verdict"] == "success"]
        ok81 = [r for r in ok if r["task_id"][:8] not in UNSOLVABLE]
        lines.append(f"| {agent} | {strategy} | {len(rows)} | {len(ok)} ({len(ok) / 95:.1%}) "
                     f"| {len(ok81)} ({len(ok81) / 81:.1%}) |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="judge at most N new trajectories (cost sample)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--variant", choices=["original", "status"], default="original")
    a = p.parse_args()
    load_env_file(Path(os.environ.get("TOOLSCOPE_ENV_FILE", ".env")))
    tasks = {t["task_id"]: t for t in json.loads(ANNOTATIONS.read_text())}
    tools = tool_map()
    done = judged(a.variant)
    todo = [(agent, strategy, tid, rec) for (agent, strategy), folders in RUNS.items()
            for tid, rec in sorted(load(agent, strategy, folders).items()) if (agent, strategy, tid) not in done]
    if a.limit:
        todo = todo[:: max(1, len(todo) // a.limit)][: a.limit]  # spread the sample over the runs
    lock, local = threading.Lock(), threading.local()
    totals = {"in": 0, "out": 0, "n": 0, "errors": 0}

    def work(item):
        agent, strategy, tid, rec = item
        if not hasattr(local, "chat"):
            local.usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
            local.chat = openai_chat(MODEL, local.usage, base_url=os.environ["TOOLSCOPE_API_BASE_URL"],
                                     api_key=os.environ["TOOLSCOPE_API_KEY"], temperature=TEMPERATURE)
        task = tasks[tid]
        before = (local.usage["input_tokens"], local.usage["output_tokens"])
        try:
            text = local.chat(judge_messages(task["Question"], task["Annotator Metadata"]["Steps"], rec, tools,
                                             with_status=a.variant == "status"), None)["content"]
            verdict, thoughts = parse_verdict(text)
        except Exception as e:  # one failed call must not stop the run; it is retried on the next run
            log.error("judge call failed", extra={"task_id": tid, "agent": agent, "error": str(e)[:300]})
            text, verdict, thoughts = str(e)[:300], "error", ""
        tin = local.usage["input_tokens"] - before[0]
        tout = local.usage["output_tokens"] - before[1]
        row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "agent": agent,
               "strategy": strategy, "task_id": tid, "model": MODEL, "variant": a.variant, "temperature": TEMPERATURE,
               "verdict": verdict, "thoughts": thoughts, "raw": text, "input_tokens": tin, "output_tokens": tout}
        with lock:
            with OUT.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            totals["in"] += tin
            totals["out"] += tout
            totals["n"] += 1
            totals["errors"] += verdict == "error"

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(work, todo))
    done = judged(a.variant)
    SUMMARY[a.variant].write_text(summarise(done, a.variant))
    SUMMARY[a.variant].with_suffix(".json").write_text(json.dumps([
        {"agent": ag, "strategy": st, "judged": sum(1 for k in done if k[:2] == (ag, st)),
         "success": sum(1 for k, r in done.items() if k[:2] == (ag, st) and r["verdict"] == "success"),
         "success_solvable": sum(1 for k, r in done.items() if k[:2] == (ag, st) and r["verdict"] == "success"
                                 and k[2][:8] not in UNSOLVABLE)} for ag, st in RUNS], indent=1) + "\n")
    cost = totals["in"] / 1e6 * RATE_IN + totals["out"] / 1e6 * RATE_OUT
    print(f"judged_now={totals['n']} errors={totals['errors']} tokens_in={totals['in']} tokens_out={totals['out']} "
          f"cost_inr≈{cost:.1f} total_judged={len(done)} out={OUT}")


if __name__ == "__main__":
    main()

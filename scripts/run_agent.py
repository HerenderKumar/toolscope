"""Run the agent (an LLM + the gateway's two tools) over tasks; one log line per task in logs/agent.jsonl.

Run (3 tasks, local 7B, execute mode):
  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/run_agent.py --strategy semantic --limit 3
Paid comparison (D-021; needs TOOLSCOPE_API_BASE_URL and TOOLSCOPE_API_KEY in .env):
  ... scripts/run_agent.py --backend openai --model <model-name> --limit 3
Options: --mode execute|selection, --k, --max-steps, --tasks <task_id prefix ...>
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from toolscope.agent.backends import (MAX_COMPLETION_TOKENS, OLLAMA_NUM_CTX, BudgetExceeded, ProviderUnavailable,
                                      ollama_chat, openai_chat)
from toolscope.agent.loop import MAX_TOOL_RESULT_CHARS, run_task
from toolscope.agent.resume import completed_task_ids, network_ok, wait_for_network
from toolscope.gateway.server import load_env_file
from toolscope.log import get_logger

log = get_logger("toolscope.run_agent")

GOLD = Path(os.environ.get("TOOLSCOPE_GOLD", "data/gold/gold.jsonl"))
OUT = Path(os.environ.get("TOOLSCOPE_AGENT_LOG", "logs/agent.jsonl"))
OLLAMA = os.environ.get("TOOLSCOPE_OLLAMA_URL", "http://localhost:11434")


def make_chat(a: argparse.Namespace, usage: dict[str, int]):
    if a.backend == "ollama":
        return ollama_chat(a.model, usage, base_url=OLLAMA, temperature=a.temperature)
    load_env_file(Path(os.environ.get("TOOLSCOPE_ENV_FILE", ".env")))
    base_url, key = os.environ.get("TOOLSCOPE_API_BASE_URL"), os.environ.get("TOOLSCOPE_API_KEY")
    if not base_url or not key:
        raise SystemExit("set TOOLSCOPE_API_BASE_URL and TOOLSCOPE_API_KEY in .env for --backend openai")
    return openai_chat(a.model, usage, base_url=base_url, api_key=key, temperature=a.temperature)


async def main(a: argparse.Namespace) -> None:
    tasks = [json.loads(line) for line in GOLD.read_text().splitlines()]
    if a.tasks:
        tasks = [t for t in tasks if any(t["task_id"].startswith(p) for p in a.tasks)]
    tasks = tasks[:a.limit] if a.limit else tasks
    key = {"strategy": a.strategy, "backend": a.backend, "model": a.model, "mode": a.mode, "k": a.k,
           "temperature": a.temperature}
    if not a.no_resume:
        finished = completed_task_ids(OUT, key)
        if finished:
            log.info("resuming: skipping finished tasks", extra={"ctx": {"skipped": len(finished)}})
        tasks = [t for t in tasks if t["task_id"] not in finished]
    run_id = datetime.now(timezone.utc).strftime(f"agent-{a.backend}-{a.strategy}-%Y%m%dT%H%M%SZ")
    env = {**os.environ, "TOOLSCOPE_STRATEGY": a.strategy}
    transport = StdioTransport(command=sys.executable, args=["-m", "toolscope.gateway.server"], env=env, cwd=os.getcwd())
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
    chat = make_chat(a, usage)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = errors = 0
    t0 = time.time()
    async with Client(transport) as gw:
        async def gateway(name, args):
            return (await gw.call_tool(name, args, raise_on_error=False)).structured_content
        for task in tasks:
            waited = wait_for_network(on_wait=lambda s: log.warning(
                "network down; waiting before next task", extra={"ctx": {"waited_s": s, "task_id": task["task_id"]}}))
            meta = {"run_id": run_id, **key, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    # D-063 settings; records without them ran with 4,000 chars / 12 steps / no reply cap
                    "max_tool_result_chars": MAX_TOOL_RESULT_CHARS, "max_steps": a.max_steps,
                    "max_completion_tokens": MAX_COMPLETION_TOKENS,
                    **({"num_ctx": OLLAMA_NUM_CTX} if a.backend == "ollama" else {})}
            usage.update(llm_calls=0, input_tokens=0, output_tokens=0, length_stops=0, max_prompt_tokens=0)
            try:
                rec = await run_task(task["task_id"], task["question"], chat, gateway, k=a.k,
                                     execute=a.mode == "execute", max_steps=a.max_steps)
                done += 1
            except BudgetExceeded as e:
                (OUT.parent / "BUDGET_EXCEEDED").write_text(f"{datetime.now(timezone.utc).isoformat()} {e}\n")
                log.error("paid API budget exceeded; stopping without recording this task",
                          extra={"ctx": {"task_id": task["task_id"]}})
                print(f"BUDGET_EXCEEDED after {done} tasks; marker {OUT.parent / 'BUDGET_EXCEEDED'}")
                raise SystemExit(3)
            except ProviderUnavailable as e:
                log.error("paid API unavailable after retries; stopping without recording this task (resume reruns it)",
                          extra={"ctx": {"task_id": task["task_id"], "error": str(e)[:200]}})
                print(f"PROVIDER_UNAVAILABLE after {done} tasks: {str(e)[:200]}")
                raise SystemExit(4)
            except Exception as e:
                errors += 1
                log.exception("task failed", extra={"ctx": {"task_id": task["task_id"]}})
                rec = {"task_id": task["task_id"], "steps": [], "chosen_tools": [], "final_answer": None,
                       "stop_reason": f"error: {type(e).__name__}: {e}"[:500], "n_steps": 0, "latency_ms": None}
            # network checked after the task too: a drop during the task marks it for re-running on resume
            net = {"network_ok": network_ok(), "network_waited_s": waited}
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**meta, **rec, **usage, **net}, ensure_ascii=False) + "\n")
            log.info("task done", extra={"ctx": {"task_id": task["task_id"], "stop": rec["stop_reason"],
                                                 "calls": len(rec["chosen_tools"]), "ms": rec["latency_ms"]}})
    print(f"tasks={len(tasks)} done={done} errors={errors} strategy={a.strategy} model={a.model} "
          f"mode={a.mode} seconds={time.time() - t0:.0f} run_id={run_id} out={OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default=os.environ.get("TOOLSCOPE_STRATEGY", "semantic"))
    p.add_argument("--model", default=os.environ.get("TOOLSCOPE_AGENT_MODEL", "qwen2.5:7b-instruct"))
    p.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    p.add_argument("--mode", choices=["execute", "selection"], default="execute")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TOOLSCOPE_TEMPERATURE", "0.7")),
                   help="LLM sampling temperature (LiveMCPBench uses 0.7; D-039)")
    p.add_argument("--max-steps", type=int, default=int(os.environ.get("TOOLSCOPE_MAX_STEPS", "30")),
                   help="safety cap; LiveMCPBench has none (D-063); 12 before 31 Oct")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--no-resume", action="store_true", help="re-run tasks already finished for this configuration")
    asyncio.run(main(p.parse_args()))

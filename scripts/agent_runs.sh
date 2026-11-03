#!/usr/bin/env bash
# Run the agent over all tasks for each given strategy, each from the same clean container (D-023).
# Usage: bash scripts/agent_runs.sh bm25 semantic        (extra run_agent.py options via AGENT_ARGS)
set -uo pipefail
cd "$(dirname "$0")/.."

IMAGE="${TOOLSCOPE_CLEAN_IMAGE:-toolscope/livemcpbench-clean:20261031a}"
CONTAINER="${TOOLSCOPE_CONTAINER:-LiveMCPBench_container}"

reset_container() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1
  docker run -itd --platform linux/amd64 -v "$PWD/third_party/LiveMCPBench:/outside:ro" --ipc=host --net=host \
    --name "$CONTAINER" "$IMAGE" bash >/dev/null
  sleep 5
  docker ps --filter "name=$CONTAINER" --format '{{.Names}} {{.Status}}'
}

for strategy in "$@"; do
  echo "== $(date -u +%FT%TZ) strategy=$strategy: resetting container"
  reset_container
  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/run_agent.py --strategy "$strategy" ${AGENT_ARGS:-} 2>>logs/agent_runs.stderr.log
  echo "== $(date -u +%FT%TZ) strategy=$strategy: exit=$?"
done

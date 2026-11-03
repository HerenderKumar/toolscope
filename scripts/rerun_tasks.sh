#!/usr/bin/env bash
# Re-run chosen tasks (--no-resume) in a fresh container from a clean snapshot; records go to <log dir>/agent.jsonl,
# where the latest record per task wins. Used for D-053 (7B, free) and D-058 (paid Qwen3-235B 404 tasks).
# Usage: bash scripts/rerun_tasks.sh <strategy> <task_id> [<task_id> ...]   (full task ids)
#   TOOLSCOPE_CLEAN_IMAGE=...   snapshot (default 20261031a)
#   RERUN_LOG_DIR=...           log folder (default logs/rerun = the 7B's)
#   RERUN_CONTAINER=...         container name (default ts_rerun_manual; never the supervisor's ts_rerun)
#   AGENT_ARGS="..."            run_agent.py agent arguments (default: the 7B, --temperature 0); run_agent loads .env
set -uo pipefail
cd "$(dirname "$0")/.."
IMAGE="${TOOLSCOPE_CLEAN_IMAGE:-toolscope/livemcpbench-clean:20261031a}"
LOGDIR="${RERUN_LOG_DIR:-logs/rerun}"
CTR="${RERUN_CONTAINER:-ts_rerun_manual}"
read -r -a ARGS <<< "${AGENT_ARGS:---temperature 0}"
strategy=$1; shift
mkdir -p "$LOGDIR"
docker rm -f "$CTR" >/dev/null 2>&1
docker run -itd --platform linux/amd64 -v "$PWD/third_party/LiveMCPBench:/outside:ro" --ipc=host --net=host \
  --name "$CTR" "$IMAGE" bash >/dev/null
sleep 5
before=$(wc -l < "$LOGDIR/agent.jsonl" 2>/dev/null || echo 0)
echo "$(date -u +%FT%TZ) START $strategy $* (image $IMAGE, log $LOGDIR, args ${ARGS[*]})"
TOOLSCOPE_CONTAINER="$CTR" TOOLSCOPE_LOG_DIR="$LOGDIR" TOOLSCOPE_AGENT_LOG="$LOGDIR/agent.jsonl" \
  HF_HUB_OFFLINE=1 PYTHONPATH=src caffeinate -i -s .venv/bin/python scripts/run_agent.py \
  --strategy "$strategy" "${ARGS[@]}" --no-resume --tasks "$@" >> "$LOGDIR/stderr.log" 2>&1
code=$?
docker rm -f "$CTR" >/dev/null 2>&1
echo "$(date -u +%FT%TZ) END strategy=$strategy exit=$code new_records=$(( $(wc -l < "$LOGDIR/agent.jsonl") - before ))"

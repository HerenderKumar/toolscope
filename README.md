# ToolScope

ToolScope is an interactive workbench for inspecting Model Context Protocol (MCP) tool discovery.
It connects retrieval comparisons, chronological agent traces and capability-level evidence, and measures
how frozen description edits change tool ranks and coverage. It accompanies the ECIR 2027 demo paper
*ToolScope: Inspecting MCP Tool Discovery and Description-Edit Tradeoffs* (source in `paper/`).

- **Data:** a fixed LiveMCPBench snapshot (commit `36a51a1`): 69 servers, 525 tools, 95 tasks.
- **Six retrievers:** BM25, BGE-base semantic, ToolRet-trained E5, hybrid (RRF), cross-encoder rerank, server-first.
- **Views:** Compare (retrieval metrics), Failures (capability-level evidence per agent run),
  Tool health with a what-if description editor (computed in memory; nothing is written).
- Diagnostic categories are rule-based hypotheses for investigation; they have not been independently validated.

## Repository layout

| Path | Contents |
|---|---|
| `src/toolscope/` | Catalog, retrievers, evaluation, diagnosis, judge, gateway (FastMCP) and read-only API (FastAPI) |
| `scripts/` | Pipeline steps: build indexes, run retrieval, evaluate, run agents, judge, diagnose, repair study, figures |
| `ui/` | React (Vite) workbench |
| `data/catalog/`, `data/gold/` | Tool catalog derived from LiveMCPBench and the frozen answer key (`gold-v1`) |
| `data/human/human_labels.csv` | Author task-success labels used to validate the judge |
| `results/` | Saved measurements that the API, UI and paper read |
| `paper/` | LNCS manuscript source |
| `tests/` | pytest suite |

## Setup

Python 3.11, CPU only.

```sh
python3.11 -m venv .venv
.venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
git clone https://github.com/icip-cas/LiveMCPBench.git third_party/LiveMCPBench
git -C third_party/LiveMCPBench checkout 36a51a1
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Rebuild the local inputs that are not stored in the repository (models are free and run on CPU):

```sh
PYTHONPATH=src .venv/bin/python scripts/prefetch_models.py   # Hugging Face models into the local cache
PYTHONPATH=src .venv/bin/python scripts/build_indexes.py     # indexes/ (read-only at runtime)
PYTHONPATH=src .venv/bin/python scripts/run_retrieval.py     # logs/retrieval.jsonl
PYTHONPATH=src .venv/bin/python scripts/evaluate.py          # results/table.md, metrics.json
```

## Run the workbench

```sh
mkdir -p docker/wheels
.venv/bin/pip download torch==2.5.1 --no-deps --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu \
  --platform manylinux2014_aarch64 --python-version 3.11 -d docker/wheels
docker compose up -d --build api ui
```

The image installs the CPU PyTorch wheel from `docker/wheels/`. The command above fetches the ARM64 wheel
(Apple Silicon, ARM Linux), the platform this setup was tested on. On x86_64 the CPU wheel is named
`torch-2.5.1+cpu-cp311-cp311-linux_x86_64.whl`, so the wheel step in `docker/python.Dockerfile` must be adjusted.

Open http://localhost:5173. The API serves only saved files from `results/` and `logs/`.

**Not included:** raw agent trajectories (`logs/`), built indexes and model caches. Views that read agent
trajectories need those logs; re-running agents requires the LiveMCPBench container and LLM API access.
The what-if editor, the retrieval comparison and the saved result reports work from the files above.

## Data and licences

`data/catalog/tools.jsonl` is derived from [LiveMCPBench](https://github.com/icip-cas/LiveMCPBench)
(Apache License 2.0; copy in `data/LIVEMCPBENCH_LICENSE`). `paper/llncs.cls` and `paper/splncs04.bst` are the
unmodified Springer LNCS class files (`paper/LNCS_LICENSE.md`).

## Use of AI tools

Claude (Anthropic) helped implement the software and edit the paper. In the study, DeepSeek V3.2 judged task
success and wrote the query-blind description rewrites; a local Qwen2.5-7B produced translation controls.

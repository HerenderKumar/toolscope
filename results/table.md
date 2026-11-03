## Retrieval results (95 queries)

Strict metrics (R, nDCG, MRR) use `qrels_strict.json`; C@k and CapR@10 use the equivalence classes. † = differs from bm25 at α = 0.05 after Bonferroni correction (5 comparisons, paired t-test). Tokens@10 = mean Qwen2.5 tokens of the top-10 tool schemas; latency = median per query, CPU.

| Strategy | R@5 | R@10 | nDCG@10 | MRR@10 | C@5 | C@10 | CapR@10 | Tokens@10 | Latency ms |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.347 | 0.418 | 0.347 | 0.444 | 0.189 | 0.232 | 0.437 | 2346 | 0.1 |
| semantic | 0.417 | 0.504 | 0.423 | 0.523 | 0.200 | 0.274 | 0.524 | 2452 | 28.1 |
| toolret | 0.381 | 0.462 | 0.376 | 0.469 | 0.189 | 0.242 | 0.471 | 1842 | 27.5 |
| hybrid | 0.358 | 0.458 | 0.376 | 0.475 | 0.168 | 0.253 | 0.476 | 2153 | 36.5 |
| rerank | 0.345 | 0.430 | 0.347 | 0.427 | 0.168 | 0.232 | 0.446 | 2029 | 544.3 |
| server_first | 0.375 | 0.439 | 0.359 | 0.423 | 0.232 | 0.295 | 0.465 | 1792 | 30.9 |

## Differences vs bm25 (Bonferroni-corrected p)

| Strategy | ΔR@5 (p) | ΔR@10 (p) | ΔnDCG@10 (p) | ΔMRR@10 (p) | ΔC@5 (p) | ΔC@10 (p) | ΔCapR@10 (p) |
|---|---|---|---|---|---|---|---|
| semantic | +0.070 (0.326) | +0.086 (0.105) | +0.076 (0.066) | +0.079 (0.243) | +0.011 (1.000) | +0.042 (1.000) | +0.086 (0.118) |
| toolret | +0.033 (1.000) | +0.044 (1.000) | +0.029 (1.000) | +0.024 (1.000) | +0.000 (1.000) | +0.011 (1.000) | +0.034 (1.000) |
| hybrid | +0.011 (1.000) | +0.039 (0.361) | +0.029 (0.403) | +0.031 (0.955) | -0.021 (1.000) | +0.021 (1.000) | +0.038 (0.600) |
| rerank | -0.002 (1.000) | +0.012 (1.000) | -0.000 (1.000) | -0.018 (1.000) | -0.021 (1.000) | +0.000 (1.000) | +0.009 (1.000) |
| server_first | +0.028 (1.000) | +0.021 (1.000) | +0.012 (1.000) | -0.021 (1.000) | +0.042 (1.000) | +0.063 (0.905) | +0.027 (1.000) |

## Ablation: ToolRet-trained-e5 configuration (D-017)

Main table: configuration of ToolRet's evaluation code (mean pooling, `Instruct: \nquery: `). Row below: configuration of ToolRet's training script (CLS pooling, `Represent this sentence…`), from `logs/ablation_toolret_cls.jsonl`.

| Configuration | R@5 | R@10 | nDCG@10 | MRR@10 | C@5 | C@10 | CapR@10 |
|---|---|---|---|---|---|---|---|
| toolret (train.sh: CLS) | 0.190 | 0.291 | 0.216 | 0.263 | 0.063 | 0.137 | 0.301 |

Source: `logs/retrieval.jsonl` (run_id retrieval-20261028T162822Z).

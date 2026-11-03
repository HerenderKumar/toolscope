# Description repair study v1 (D-080)

Exploratory, selected-tool case study on original task questions. Discovery is measured; task success is not.
Translations use local Qwen plus assistant corrections; structured texts reuse DeepSeek D-073 outputs. Their difference cannot isolate a prompt/model effect. No held-out generalization or significance claim.

## Joint deployment: each complete edit set measured together

| Condition / edit set | Retriever | k | Capabilities / 255 | Complete tasks / 95 | Gained / lost capabilities | Newly / lost complete tasks | Mean context tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| original | bm25 | 5 | 80 | 18 | 0 / 0 | 0 / 0 | 1256.1 |
| original | bm25 | 10 | 98 | 22 | 0 / 0 | 0 / 0 | 2345.7 |
| original | semantic | 5 | 97 | 19 | 0 / 0 | 0 / 0 | 1226.4 |
| original | semantic | 10 | 119 | 26 | 0 / 0 | 0 / 0 | 2452.2 |
| structured (same 12 tools) | bm25 | 5 | 84 | 19 | 5 / 1 | 1 / 0 | 1221.8 |
| structured (same 12 tools) | bm25 | 10 | 102 | 25 | 6 / 2 | 3 / 0 | 2350.9 |
| structured (same 12 tools) | semantic | 5 | 101 | 21 | 8 / 4 | 2 / 0 | 1255.9 |
| structured (same 12 tools) | semantic | 10 | 126 | 27 | 7 / 0 | 1 / 0 | 2453.1 |
| structured | bm25 | 5 | 93 | 21 | 17 / 4 | 3 / 0 | 1228.6 |
| structured | bm25 | 10 | 116 | 27 | 22 / 4 | 5 / 0 | 2319.5 |
| structured | semantic | 5 | 104 | 21 | 14 / 7 | 2 / 0 | 1232.7 |
| structured | semantic | 10 | 129 | 27 | 13 / 3 | 2 / 1 | 2444.1 |
| translation | bm25 | 5 | 82 | 18 | 4 / 2 | 0 / 0 | 1224.8 |
| translation | bm25 | 10 | 103 | 22 | 6 / 1 | 0 / 0 | 2351.5 |
| translation | semantic | 5 | 101 | 20 | 6 / 2 | 1 / 0 | 1217.9 |
| translation | semantic | 10 | 127 | 27 | 8 / 0 | 1 / 0 | 2419.1 |

Capabilities use frozen acceptable-tool equivalence classes. Complete-task discovery requires every capability.
Context uses the existing Qwen2.5-3B tokenizer and schema serialization, including unchanged input schemas.
The translation set changes 12 descriptions; the full structured set contains 43. The matched structured set edits the same 12 tools as translation. English translation controls are identity edits.

## Historical candidate probes (separate denominator)

| Retriever | Improved | Unchanged | Worsened | Candidate-capability rescues @5 | Rescues @10 |
|---|---:|---:|---:|---:|---:|
| bm25 | 49 | 37 | 24 | 11 / 95 | 18 / 95 |
| semantic | 66 | 3 | 41 | 6 / 95 | 9 / 95 |

The historical 110 rows share tasks/tools. A rescued capability is not a successful task. Do not add separate single-edit gains to estimate the joint catalog.

## Four-cell repair choice example (historical wuwa probe)

| Description | BM25 rank | Semantic rank |
|---|---:|---:|
| original | 228 | 130 |
| translation | 228 | 10 |
| structured | 22 | 3 |

Tool: `wuwa-mcp/get_character_info`; task: `29fa007d-5caf-41a0-b107-e9388305c6c5`. tradeoffs.jsonl holds every selected-tool/task combination, including failures and regressions.

## Faithfulness and limits

Assistant card review flagged concerns in 22 of 43 historical rewrites. The original texts remain frozen for measurement; this is not approval to deploy them.
Six of 12 raw local translations needed corrections before scoring. Raw generations and all corrections are saved.
No author/human annotation or live execution validation was performed in this review.
Per-tool length and per-query context costs, complete-task losses and collateral capability losses are exported.
Actual agent-query repair, multilingual retrieval and stochastic end-to-end agent reruns remain untested.

Controls: 190 saved-baseline comparisons; 190 unchanged-description query/strategy comparisons; zero mismatches.

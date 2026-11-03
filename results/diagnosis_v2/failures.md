## Whose fault? Failure breakdown

Agent runs: one label per unsolved capability (D-010) in **failed** tasks (human labels where they exist, otherwise the judge; the judge is not validated for Kimi, D-075); share of all such labels in brackets.

| Agent | Strategy | Failed tasks | Environment | Agent selection | Composition | Ranking confusion | Bad description | Retrieval miss |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B (local) | bm25 | 83 | 18 (9%) | 46 (24%) | 5 (3%) | 47 (24%) | 13 (7%) | 64 (33%) |
| Qwen2.5-7B (local) | semantic | 77 | 17 (9%) | 48 (27%) | 5 (3%) | 27 (15%) | 8 (4%) | 74 (41%) |
| Qwen3-235B | bm25 | 65 | 16 (11%) | 22 (16%) | 9 (6%) | 28 (20%) | 11 (8%) | 55 (39%) |
| Qwen3-235B | semantic | 60 | 17 (14%) | 26 (21%) | 5 (4%) | 16 (13%) | 6 (5%) | 55 (44%) |
| Sonnet 5.5 | bm25 | 46 | 12 (16%) | 13 (17%) | 8 (10%) | 6 (8%) | 6 (8%) | 32 (42%) |
| Kimi K2.6 | bm25 | 57 | 16 (14%) | 23 (20%) | 10 (9%) | 21 (18%) | 5 (4%) | 39 (34%) |
| Kimi K2.6 | semantic | 41 | 12 (14%) | 20 (24%) | 8 (10%) | 6 (7%) | 2 (2%) | 36 (43%) |

Retrieval only (task question, top-10; 255 capabilities per strategy):

| Strategy | Found | Ranking confusion | Bad description | Retrieval miss |
|---|---|---|---|---|
| bm25 | 98 | 72 | 14 | 71 |
| semantic | 119 | 44 | 8 | 84 |
| toolret | 107 | 57 | 0 | 91 |
| hybrid | 108 | 62 | 0 | 85 |
| rerank | 101 | 69 | 0 | 85 |
| server_first | 101 | 63 | 0 | 91 |

D-078: description-repairable labels require a same-strategy rescue on the original task question at top-10. BM25 and semantic were tested; ToolRet, hybrid, rerank and server-first were not tested for repair. A zero in their description column is not evidence of no effect.
Agent rows use an offline task-question proxy; actual agent searches/top-5 were not replayed.
Legacy D-073 probe linkage is reconstructed from saved inputs, not newly measured.

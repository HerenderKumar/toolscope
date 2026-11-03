## Whose fault? Failure breakdown

Agent runs: one label per unsolved capability (D-010) in **failed** tasks (human labels where they exist, otherwise the judge; the judge is not validated for Kimi, D-075); share of all such labels in brackets.

| Agent | Strategy | Failed tasks | Environment | Agent selection | Composition | Ranking confusion | Bad description | Retrieval miss |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B (local) | bm25 | 86 | 18 (9%) | 46 (23%) | 7 (4%) | 49 (25%) | 14 (7%) | 65 (33%) |
| Qwen2.5-7B (local) | semantic | 80 | 17 (9%) | 48 (27%) | 5 (3%) | 27 (15%) | 8 (4%) | 76 (42%) |
| Qwen3-235B | bm25 | 67 | 15 (11%) | 22 (16%) | 9 (6%) | 27 (19%) | 10 (7%) | 57 (41%) |
| Qwen3-235B | semantic | 59 | 17 (14%) | 26 (21%) | 6 (5%) | 13 (11%) | 5 (4%) | 56 (46%) |
| Sonnet 5.5 | bm25 | 47 | 13 (16%) | 14 (18%) | 8 (10%) | 9 (11%) | 6 (8%) | 30 (38%) |
| Kimi K2.6 | bm25 | 57 | 16 (14%) | 23 (20%) | 10 (9%) | 21 (18%) | 5 (4%) | 39 (34%) |
| Kimi K2.6 | semantic | 45 | 14 (16%) | 18 (20%) | 9 (10%) | 6 (7%) | 4 (4%) | 38 (43%) |

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

## Whose fault? Failure breakdown

Agent runs: one label per unsolved capability (D-010) in **failed** tasks (human labels where they exist, otherwise the judge; the judge is not validated for Kimi, D-075); share of all such labels in brackets.

| Agent | Strategy | Failed tasks | Environment | Agent selection | Composition | Ranking confusion | Bad description | Retrieval miss |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B (local) | bm25 | 86 | 18 (9%) | 46 (23%) | 7 (4%) | 49 (25%) | 19 (10%) | 60 (30%) |
| Qwen2.5-7B (local) | semantic | 80 | 17 (9%) | 48 (27%) | 5 (3%) | 27 (15%) | 20 (11%) | 64 (35%) |
| Qwen3-235B | bm25 | 67 | 15 (11%) | 22 (16%) | 9 (6%) | 27 (19%) | 15 (11%) | 52 (37%) |
| Qwen3-235B | semantic | 59 | 17 (14%) | 26 (21%) | 6 (5%) | 13 (11%) | 11 (9%) | 50 (41%) |
| Sonnet 5.5 | bm25 | 47 | 13 (16%) | 14 (18%) | 8 (10%) | 9 (11%) | 9 (11%) | 27 (34%) |
| Kimi K2.6 | bm25 | 57 | 16 (14%) | 23 (20%) | 10 (9%) | 21 (18%) | 7 (6%) | 37 (32%) |
| Kimi K2.6 | semantic | 45 | 14 (16%) | 18 (20%) | 9 (10%) | 6 (7%) | 6 (7%) | 36 (40%) |

Retrieval only (task question, top-10; 255 capabilities per strategy):

| Strategy | Found | Ranking confusion | Bad description | Retrieval miss |
|---|---|---|---|---|
| bm25 | 98 | 72 | 19 | 66 |
| semantic | 119 | 44 | 22 | 70 |
| toolret | 107 | 57 | 20 | 71 |
| hybrid | 108 | 62 | 18 | 67 |
| rerank | 101 | 69 | 17 | 68 |
| server_first | 101 | 63 | 22 | 69 |

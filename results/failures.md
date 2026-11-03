## Whose fault? Failure breakdown

Agent runs: one label per unsolved capability (D-010) in **failed** tasks (human labels where they exist, otherwise the judge; the judge is not validated for Kimi, D-075); share of all such labels in brackets.

| Agent | Strategy | Failed tasks | Environment | Agent selection | Composition | Ranking confusion | Bad description | Retrieval miss |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B (local) | bm25 | 83 | 18 (9%) | 46 (24%) | 5 (3%) | 47 (24%) | 17 (9%) | 60 (31%) |
| Qwen2.5-7B (local) | semantic | 77 | 17 (9%) | 48 (27%) | 5 (3%) | 27 (15%) | 20 (11%) | 62 (35%) |
| Qwen3-235B | bm25 | 65 | 16 (11%) | 22 (16%) | 9 (6%) | 28 (20%) | 16 (11%) | 50 (35%) |
| Qwen3-235B | semantic | 60 | 17 (14%) | 26 (21%) | 5 (4%) | 16 (13%) | 12 (10%) | 49 (39%) |
| Sonnet 5.5 | bm25 | 46 | 12 (16%) | 13 (17%) | 8 (10%) | 6 (8%) | 7 (9%) | 31 (40%) |
| Kimi K2.6 | bm25 | 57 | 16 (14%) | 23 (20%) | 10 (9%) | 21 (18%) | 7 (6%) | 37 (32%) |
| Kimi K2.6 | semantic | 41 | 12 (14%) | 20 (24%) | 8 (10%) | 6 (7%) | 3 (4%) | 35 (42%) |

Retrieval only (task question, top-10; 255 capabilities per strategy):

| Strategy | Found | Ranking confusion | Bad description | Retrieval miss |
|---|---|---|---|---|
| bm25 | 98 | 72 | 19 | 66 |
| semantic | 119 | 44 | 22 | 70 |
| toolret | 107 | 57 | 20 | 71 |
| hybrid | 108 | 62 | 18 | 67 |
| rerank | 101 | 69 | 17 | 68 |
| server_first | 101 | 63 | 22 | 69 |

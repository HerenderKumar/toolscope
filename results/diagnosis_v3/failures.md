## Whose fault? Failure breakdown

Agent runs: D-079 labels for unsolved capabilities with sufficient recorded evidence in failed tasks. Shares use assigned labels only; unassigned capabilities are reported separately below. Human labels override the judge; Kimi semantic outcomes remain exploratory (D-075).

| Agent | Strategy | Failed tasks | Environment | Agent selection | Composition | Ranking confusion | Bad description | Retrieval miss |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B (local) | bm25 | 83 | 18 (9%) | 22 (11%) | 0 (0%) | 56 (29%) | 0 (0%) | 96 (50%) |
| Qwen2.5-7B (local) | semantic | 77 | 17 (10%) | 26 (15%) | 0 (0%) | 33 (19%) | 0 (0%) | 94 (55%) |
| Qwen3-235B | bm25 | 65 | 16 (12%) | 11 (8%) | 0 (0%) | 51 (38%) | 0 (0%) | 55 (41%) |
| Qwen3-235B | semantic | 60 | 17 (14%) | 9 (8%) | 0 (0%) | 34 (29%) | 0 (0%) | 58 (49%) |
| Sonnet 5.5 | bm25 | 46 | 12 (16%) | 10 (13%) | 0 (0%) | 18 (24%) | 0 (0%) | 36 (47%) |
| Kimi K2.6 | bm25 | 57 | 16 (15%) | 15 (14%) | 0 (0%) | 24 (23%) | 0 (0%) | 49 (47%) |
| Kimi K2.6 | semantic | 41 | 12 (15%) | 18 (22%) | 0 (0%) | 9 (11%) | 0 (0%) | 41 (51%) |

Retrieval only (task question, top-10; 255 capabilities per strategy):

| Strategy | Found | Ranking confusion | Bad description | Retrieval miss |
|---|---|---|---|---|
| bm25 | 98 | 72 | 14 | 71 |
| semantic | 119 | 44 | 8 | 84 |
| toolret | 107 | 57 | 0 | 91 |
| hybrid | 108 | 62 | 0 | 85 |
| rerank | 101 | 69 | 0 | 85 |
| server_first | 101 | 63 | 0 | 91 |

D-079: agent rows use observed query/cutoff and chronological exposure; offline rows retain D-078.
Unassigned capabilities are coverage limits, not a seventh category. Actual-query description repair and dependency failures were not tested.
Offline repair probes exist for BM25 and semantic only; zero description labels for the other four retrievers mean not tested.
Recorded event order is available; assistant-turn grouping is not. Query adequacy is not judged.

| Agent | Strategy | Unassigned capabilities in failed tasks |
|---|---|---:|
| Qwen2.5-7B (local) | bm25 | 1 |
| Qwen2.5-7B (local) | semantic | 9 |
| Qwen3-235B | bm25 | 8 |
| Qwen3-235B | semantic | 7 |
| Sonnet 5.5 | bm25 | 1 |
| Kimi K2.6 | bm25 | 10 |
| Kimi K2.6 | semantic | 4 |

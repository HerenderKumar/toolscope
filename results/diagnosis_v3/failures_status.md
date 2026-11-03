## Whose fault? Failure breakdown

Agent runs: D-079 labels for unsolved capabilities with sufficient recorded evidence in failed tasks. Shares use assigned labels only; unassigned capabilities are reported separately below. Human labels override the judge; Kimi semantic outcomes remain exploratory (D-075).

| Agent | Strategy | Failed tasks | Environment | Agent selection | Composition | Ranking confusion | Bad description | Retrieval miss |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B (local) | bm25 | 86 | 18 (9%) | 23 (12%) | 0 (0%) | 60 (30%) | 0 (0%) | 97 (49%) |
| Qwen2.5-7B (local) | semantic | 80 | 17 (10%) | 26 (15%) | 0 (0%) | 33 (19%) | 0 (0%) | 96 (56%) |
| Qwen3-235B | bm25 | 67 | 15 (11%) | 11 (8%) | 0 (0%) | 51 (39%) | 0 (0%) | 55 (42%) |
| Qwen3-235B | semantic | 59 | 17 (15%) | 9 (8%) | 0 (0%) | 32 (28%) | 0 (0%) | 58 (50%) |
| Sonnet 5.5 | bm25 | 47 | 13 (16%) | 12 (15%) | 0 (0%) | 21 (27%) | 0 (0%) | 33 (42%) |
| Kimi K2.6 | bm25 | 57 | 16 (15%) | 15 (14%) | 0 (0%) | 24 (23%) | 0 (0%) | 49 (47%) |
| Kimi K2.6 | semantic | 45 | 14 (16%) | 16 (19%) | 0 (0%) | 10 (12%) | 0 (0%) | 45 (53%) |

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

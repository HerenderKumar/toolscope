# Diagnosis v2 change audit (D-078)

Same frozen inputs; only description-repair evidence scope changes.

| Scope / run | Strategy | v1 description labels | v2 description labels |
|---|---|---:|---:|
| Qwen2.5-7B (local) | bm25 | 19 | 14 |
| Qwen2.5-7B (local) | semantic | 20 | 8 |
| Qwen3-235B | bm25 | 17 | 12 |
| Qwen3-235B | semantic | 15 | 8 |
| Sonnet 5.5 | bm25 | 11 | 8 |
| Kimi K2.6 | bm25 | 11 | 7 |
| Kimi K2.6 | semantic | 12 | 7 |
| retrieval | bm25 | 19 | 14 |
| retrieval | semantic | 22 | 8 |
| retrieval | toolret | 20 | 0 |
| retrieval | hybrid | 18 | 0 |
| retrieval | rerank | 17 | 0 |
| retrieval | server_first | 22 | 0 |

Changed labels: 137 of 3315. See changes.json for each affected capability.
Agent counts above include all tasks, not only failed tasks; the failure reports apply the unchanged outcome policy.
The four unprobed strategies are not tested for repair. Their zero description labels are not a negative experimental result.
Agent repair evidence is an offline task-question/top-10 proxy. Actual agent searches are not replayed.
Existing probes and cached descriptions are reused; query/intervention linkage is reconstructed, not newly measured.

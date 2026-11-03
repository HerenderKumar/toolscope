# Diagnosis v3: actual agent-search evidence (D-079)

Original logged shortlists establish returned tools; only untruncated responses establish observed exposure.
Recorded event order is available, but assistant-turn grouping is not. Earlier exposure is not proof of reasoning.
Replay compares retrievers on the exact logged query and k. It does not judge query adequacy or replay descriptions.

| Agent | Strategy | Assigned unsolved capabilities | Unassigned | Acceptable call succeeded |
|---|---|---:|---:|---:|
| Qwen2.5-7B (local) | bm25 | 207 | 1 | 47 |
| Qwen2.5-7B (local) | semantic | 183 | 13 | 59 |
| Qwen3-235B | bm25 | 153 | 8 | 94 |
| Qwen3-235B | semantic | 146 | 7 | 102 |
| Sonnet 5.5 | bm25 | 127 | 2 | 126 |
| Kimi K2.6 | bm25 | 141 | 10 | 104 |
| Kimi K2.6 | semantic | 128 | 9 | 118 |

Counts above include all tasks. Failure reports separately apply unchanged human/judge task outcomes.
Changed labels: 401; details in changes.json. Null labels are coverage limitations, not a seventh fault category.
V2 task-question diagnoses remain nested as offline context. No actual-query composition or description-repair claim is inferred from them.

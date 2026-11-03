## Agent runs (D-063 settings, final record per task)

Gold tool used = at least one *successful* call to an acceptable tool (gold-v1 classes); all capabilities = every capability of the task got a successful acceptable call. These are tool-use measures, not task success (LiveMCPEval comes in Phase 5). Cost covers final records only.

| Agent | Strategy | Tasks | Final | Step cap | No tool call | Tasks with a tool error | Calls | Call error rate | Gold tool used | All capabilities | Tokens in / out | Cost ₹ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B (local) | bm25 | 95 | 95 | 0 | 21 | 36 | 241 | 37% | 33 | 6 | 1.29M / 0.06M | 0 |
| Qwen2.5-7B (local) | semantic | 95 | 94 | 1 | 17 | 33 | 290 | 26% | 44 | 8 | 1.68M / 0.06M | 0 |
| Qwen3-235B | bm25 | 95 | 95 | 0 | 12 | 36 | 370 | 22% | 61 | 21 | 3.41M / 0.08M | 202 |
| Qwen3-235B | semantic | 95 | 92 | 3 | 7 | 32 | 471 | 12% | 61 | 23 | 5.09M / 0.09M | 295 |
| Sonnet 5.5 | bm25 | 95 | 89 | 6 | 4 | 28 | 834 | 19% | 71 | 31 | 12.85M / 0.60M | 3,578 |
| Kimi K2.6 | bm25 | 95 | 94 | 1 | 8 | 30 | 517 | 14% | 66 | 26 | 4.50M / 0.26M | 438 |
| Kimi K2.6 | semantic | 95 | 93 | 2 | 6 | 22 | 596 | 8% | 72 | 29 | 6.47M / 0.32M | 607 |

Final-record cost ≈ ₹5,121 (Qwen at listed prices; the aicredits dashboard is authoritative and also includes earlier, superseded attempts).

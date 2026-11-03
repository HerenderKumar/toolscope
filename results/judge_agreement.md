## Judge agreement with human task-success labels (judge deepseek/deepseek-v3.2, variant original)

Trust rule (D-072): the judge is used only if Cohen's kappa >= 0.7. LiveMCPBench reports raw agreement (DeepSeek-V3-0324: 81.05%).

| Labels | n | Raw agreement | Cohen's kappa (95% CI) | Passes | Human success | Judge success | Confusion H/J: f-f, f-s, s-f, s-s |
|---|---|---|---|---|---|---|---|
| all | 155 | 79.4% | 0.563 (0.42–0.69) | no | 51 | 65 | 81, 23, 9, 42 |
| kimi_bm25_full | 95 | 71.6% | 0.430 (0.25–0.60) | no | 38 | 47 | 39, 18, 9, 29 |
| sample60 | 60 | 91.7% | 0.784 (0.59–0.95) | yes | 13 | 18 | 42, 5, 0, 13 |

Confusion (all): human=failure/judge=failure: 81, human=failure/judge=success: 23, human=success/judge=failure: 9, human=success/judge=success: 42

## Judge agreement with human task-success labels (judge deepseek/deepseek-v3.2, variant status)

Trust rule (D-072): the judge is used only if Cohen's kappa >= 0.7. LiveMCPBench reports raw agreement (DeepSeek-V3-0324: 81.05%).

| Labels | n | Raw agreement | Cohen's kappa (95% CI) | Passes | Human success | Judge success | Confusion H/J: f-f, f-s, s-f, s-s |
|---|---|---|---|---|---|---|---|
| all | 155 | 79.4% | 0.563 (0.42–0.69) | no | 51 | 65 | 81, 23, 9, 42 |
| kimi_bm25_full | 95 | 72.6% | 0.458 (0.28–0.63) | no | 38 | 50 | 38, 19, 7, 31 |
| sample60 | 60 | 90.0% | 0.721 (0.47–0.91) | yes | 13 | 15 | 43, 4, 2, 11 |

Confusion (all): human=failure/judge=failure: 81, human=failure/judge=success: 23, human=success/judge=failure: 9, human=success/judge=success: 42

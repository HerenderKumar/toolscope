## Where the judge disagrees with the human labels (status variant, D-074; 155 labelled trajectories)

32 disagreements (Kimi bm25 26, blind sample 6), sorted by reason after reading each case (1 Nov 2026).
**No human label was changed** because of this review. Cases where the human label is debatable are listed, not edited.

### Judge says success, human says failure (23)
| Reason | n | Cases |
|---|---|---|
| Incorrect or out-of-scope answer the judge cannot check without results/ground truth (wrong date, wrong paper, wrong item, reversed exchange rate, 6 months instead of a year, invalid route) | 9 | 04282571, 4aa079dd, 5c19a347, 80bb6782, 88ccea26, be3f2275, d1ea5598, fbe982c3 (Kimi); 086e2393 (Qwen3-235B) |
| A required key point skipped (Word step, current time, security scan, APA formatting, attractions) | 5 | 17a4aef0, 470f7ac1, cf1fca1e (Kimi); 71a7a42a, 10b5dbf7 (Sonnet) |
| Facts from the agent's own knowledge; no source retrieval call, only write/create calls | 3 | 08644b1a, a984d31b (Kimi); a984d31b (Sonnet) |
| OK-but-unusable or failed result accepted (empty transcript, paper not found, the same route for every mode, failed news call) | 4 | 1a3ffa81, 7bda47e8, 8f26fc1a, c4d6c051 (Kimi) |
| Strict human reading of a key point (ticker not resolved by a tool; curl instead of the required wget) | 2 | 39c4c9fc, 4d7582c4 (Kimi) |

### Judge says failure, human says success (9)
| Reason | n | Cases |
|---|---|---|
| Judge stricter than the rule on a key point's wording, or ignores a successful recovery/alternative (separate Echo call, exact one-month window, failed side search, Context7 instead of DeepWiki, abstracts in the answer, how the chart file was saved) | 6 | 29fa007d, 38c156c5, 4cbd9682, 860f7b99, e9df60b1 (Kimi); 4de2e9f7 (Sonnet) |
| Judge error: treats Oct 2026 papers as "in the future" (its own date assumption) | 1 | fa451c3f (Kimi) |
| Judge misread: says the answer gives another person's email; the CSV row for Sheryl Meyers really holds mariokhan@ryan-pope.org, matching the reference | 1 | 507f592a (Kimi) |
| Debatable human label: Sonnet answered an English request **in Arabic** (content from Hacker News calls); the judge flagged it | 1 | 8aaa7e7f (Sonnet) |

### What this shows
- 13 of the 23 lenient errors (incorrect answers + unusable results) need the **result content or ground truth**, which LiveMCPEval never shows the judge. A status flag cannot fix them. This motivates an evidence-aware judge (results + supporting call per key point), which would need its own fresh validation.
- 8 judge errors (5 skipped key points + 3 memory answers) are visible from the inputs the judge already has: the judge under-applies the key-point and "internal knowledge" rules.
- On the strict side, the judge mostly ignores recoveries through alternative tools.

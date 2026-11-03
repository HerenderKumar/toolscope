# ToolScope: three-minute demo walkthrough

Status: script prepared against saved source data and the deployed API. Original/structured what-if ranks were
verified over live HTTP on 3 Nov 2026 (see `results/demo_http_20261103.json`). Browser interaction and recording are
not verified: the browser access host was unavailable on 3 Nov 2026. No video file or public URL exists yet.

Open http://localhost:5173. The workbench is already running. To restart the built images:

```sh
docker compose up -d --no-build api ui
```

Expected: API becomes healthy and UI starts. `docker compose ps` confirms both. Use a normal browser to check
the interactions below before recording. Browser zoom 100%; enlarge the window enough to read both panes.

## 0:00–0:25 — problem and comparison

Open Compare. Say: “ToolScope helps inspect why an MCP capability was missed. It links saved discovery traces
with retrieval comparisons and lets us measure description-edit effects.”

Point to semantic nDCG@10 0.423 versus BM25 0.347. Say: “This is a descriptive difference. None of the six
configurations differs significantly from BM25 after the five-comparison correction.”

## 0:25–1:15 — case 1: an exposed tool was not called

Open Failures; keep actual-agent-search diagnosis v3 selected. In **Qwen2.5-7B (local), bm25**, open the
**Agent selection** segment. Its saved count is 23. The first case is task
`0180954e-dc60-4df6-be68-63562fcafee2`, capability `get-douyin-trending` (#4).

Question: “Show me today's trending topics in the gaming world across all platforms.”
Expand search evidence and trajectory. The search `trending topics in gaming` returned Douyin at rank 2 in an
untruncated five-tool shortlist. The next and only call used `wikipedia/get_related_topics`, with valid arguments
`{"limit":10,"title":"gaming"}`. The gateway reported OK. No acceptable Douyin call occurred.

Say: “The required tool was exposed before this call. The rule suggests a selection problem. We can inspect
what was searched, what was called, and the saved answer. A successful call alone does not establish that this
capability was completed. These categories are investigation hypotheses, not independently validated causes.”

Check Next / Previous pagination and return to the first page before recording. Expected pages: 1–20 of 23,
then 21–23 of 23. The full final answer and raw arguments should expand without errors.

## 1:15–2:15 — case 2: a measured description edit

Open http://localhost:5173/#tool=wuwa-mcp/get_character_info.
Show the current description and saved rewrite effect. Select the **Query-blind rewrite (DeepSeek V3.2)** button,
then **Measure rank**. The first measurement may load the local embedding model; rehearse once before recording.
Use the task `29fa007d-5caf-41a0-b107-e9388305c6c5`.

Expected structured single-edit ranks: **BM25 228 → 22; semantic 130 → 3**.
Click Original and measure again: original-to-original ranks should remain 228 → 228 and 130 → 130.
Do not edit the text after seeing the task and present that new edit as a query-blind experiment.

Say: “This frozen rewrite improves semantic discovery into the top five. BM25 improves but still misses the top
ten. A separate saved translation control reaches semantic rank ten, which meets a top-ten cutoff but misses a
top-five cutoff. Discovery depends on both the retriever and the cutoff.”

The translation rank (130 → 10; BM25 remains 228) comes from `results/repair_study_v1/summary.json`, under
`example`. It is not a separate condition button in the current UI. Show the report if discussing it on screen.
The API measurement is an in-memory operation and saves no index or experimental result.

## 2:15–3:00 — case 3: gains and regressions together

Open `results/repair_study_v1/report.md` in the editor preview (separate report, not an existing UI panel).
Use the **structured / semantic / k=10** joint-set row. Complete-task discovery changes **26 → 27 out of 95**,
but this is **two newly complete tasks and one lost complete task**. Capabilities change 119 → 129, with 13 gains
and three losses. The matching BM25 joint-set row changes complete-task discovery 22 → 27.

Say: “A promising single rewrite is not enough. Joint edits change competition among tools, so we show gains
and losses. These counts measure complete discovery, not agent task success. The original catalog and indexes
remain intact.” End with the three views and the intended audience: MCP developers and retrieval researchers.

## Recording and readiness

Record the real interactions using the user's available screen recorder; do not substitute a slide video and
call it a live interaction recording. Check audio, text legibility and that loading/error states are explained.
Return the saved video path for review before publishing. A public live-demo or video URL is required by the
ECIR demo call. This local address is not a public demo URL.

Manual pass conditions: Compare values match the saved table; Failures paging matches the counts above;
evidence and final answer expand; both original and rewritten what-if measurements match the expected ranks;
no browser errors are visible. A failed check should be recorded and fixed before claiming demo readiness.

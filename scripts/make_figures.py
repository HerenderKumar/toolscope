"""Paper figures from results/ only: Fig. 3 failure breakdown (stacked shares per agent run) and the LaTeX results
table (retrieval + agents). Writes paper/figures/failures.pdf/.png and paper/figures/results_table.tex.

Palette: six categorical slots of the dataviz reference palette in a fixed order, checked with its validator
(light mode: all checks pass; contrast warning -> shares are printed on the bars in dark ink, plus a legend).

Run:  PYTHONPATH=src .venv/bin/python scripts/make_figures.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("results")
OUT = Path("paper/figures")
LABELS = ["environment failure", "agent selection error", "composition error", "ranking confusion",
          "bad description", "retrieval miss"]
NAMES = ["Environment", "Agent selection", "Composition", "Ranking confusion", "Descr.-repairable",
         "Retrieval miss"]
COLORS = ["#1baf7a", "#eb6834", "#2a78d6", "#eda100", "#e87ba4", "#008300"]
INK, MUTED = "#1d1d1f", "#5f5f63"
SHORT_AGENT = {"Qwen2.5-7B (local)": "Qwen2.5-7B", "Qwen3-235B": "Qwen3-235B", "Sonnet 5.5": "Sonnet 5.5",
               "Kimi K2.6": "Kimi K2.6"}


def label_ink(fill: str) -> str:
    """Dark ink on light fills, white on dark fills (WCAG relative luminance)."""
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(int(fill[i:i + 2], 16) / 255) for i in (1, 3, 5))
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.2 else "white"


def failure_figure(diagnosis: str = "v1") -> int:
    folder = RESULTS / "diagnosis_v2" if diagnosis == "v2" else RESULTS
    data = json.loads((folder / "failures_status.json").read_text())["agent"]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.edgecolor": MUTED,
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.8, 2.35))
    rows = list(reversed(data))
    for y, r in enumerate(rows):
        total = sum(r["labels"][l] for l in LABELS)
        left = 0.0
        for l, color in zip(LABELS, COLORS):
            share = r["labels"][l] / total
            ax.barh(y, share, left=left, height=0.68, color=color, edgecolor="white", linewidth=1.0)
            if share >= 0.07:
                ax.text(left + share / 2, y, f"{share:.0%}", ha="center", va="center", fontsize=6,
                        color=label_ink(color))
            left += share
    ylabels = []
    for r in rows:
        src = " (human)" if r.get("failed_by") == "human" else ""
        if diagnosis == "v2" and r.get("exploratory"):
            src += " (exploratory)"
        ylabels.append(f"{SHORT_AGENT[r['agent']]} · {r['strategy']}{src}\n{r['failed_tasks']} failed tasks")
    ax.set_yticks(range(len(rows)), ylabels, fontsize=6, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"], fontsize=6, color=MUTED)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    ax.legend(handles, NAMES, ncol=3, fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.42, 1.0),
              handlelength=1.0, columnspacing=1.0)
    if diagnosis == "v2":
        ax.set_xlabel("D-078: task-question/top-10 repair evidence; agent searches not replayed", fontsize=6)
    fig.tight_layout(pad=0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "failures.pdf")
    fig.savefig(OUT / "failures.png", dpi=220)
    return len(rows)


def results_table() -> int:
    m = json.loads((RESULTS / "metrics.json").read_text())
    runs = json.loads((RESULTS / "agent_runs.json").read_text())
    success = {(r["agent"], r["strategy"]): r for r in json.loads((RESULTS / "task_success_status.json").read_text())}
    judge = {g["labels"]: g for g in json.loads((RESULTS / "judge_agreement_status.json").read_text())["groups"]}
    strategies = list(m["strategies"])
    best = {k: max(m["strategies"][s]["mean"][k] for s in strategies) for k in ("recall@10", "ndcg@10", "completeness@10")}

    def cell(s: str, k: str) -> str:
        v = m["strategies"][s]["mean"][k]
        return f"\\textbf{{{v:.3f}}}" if v == best[k] else f"{v:.3f}"

    lines = [r"\begin{tabular}{@{}lrrr@{\hspace{1.2em}}llrr@{}}", r"\toprule",
             r"\multicolumn{4}{@{}l}{\textit{Retrieval (95 task questions)}} & "
             r"\multicolumn{4}{l@{}}{\textit{Agents on live MCP servers (95 tasks)}}\\",
             r"Strategy & R@10 & nDCG@10 & C@10 & Agent & Retr. & Success & Gold tool\\", r"\midrule"]
    agent_rows = []
    for r in runs:
        key = (r["agent"], r["strategy"])
        if r["agent"].startswith("Kimi") and r["strategy"] == "bm25":
            succ = f"{judge['kimi_bm25_full']['human_success'] / 95:.0%}$^h$"
        elif r["agent"].startswith("Kimi"):
            succ = "--"
        else:
            succ = f"{success[key]['success'] / 95:.0%}"
        agent_rows.append(f"{SHORT_AGENT[r['agent']]} & {r['strategy']} & {succ} & {r['gold_any'] / r['n']:.0%}")
    for i in range(max(len(strategies), len(agent_rows))):
        left = (f"{strategies[i].replace('_', chr(92) + '_')} & {cell(strategies[i], 'recall@10')} & {cell(strategies[i], 'ndcg@10')} & "
                f"{cell(strategies[i], 'completeness@10')}") if i < len(strategies) else "& & &"
        right = agent_rows[i] if i < len(agent_rows) else "& & &"
        lines.append(f"{left} & {right}\\\\".replace("%", "\\%"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "results_table.tex").write_text("\n".join(lines) + "\n")
    return len(lines)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", choices=("v1", "v2"), default="v1")
    diagnosis = parser.parse_args().diagnosis
    if diagnosis == "v2":
        OUT = OUT / "diagnosis_v2"
    bars = failure_figure(diagnosis)
    rows = results_table()
    print(f"failure_bars={bars} table_lines={rows} out={OUT}")


if __name__ == "__main__":
    main()

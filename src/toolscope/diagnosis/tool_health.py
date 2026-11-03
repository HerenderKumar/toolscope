"""Tool-health facts per tool (Phase 5 Step 4): how often the tool was missed when it was the answer, and simple
description flags (empty, short, non-English, parameters without a description). Inputs are the catalog,
the answer key and the failure labels; nothing here calls a model."""

import re
from collections import defaultdict
from typing import Any

SHORT_CHARS = 30  # well under the catalog's median description length (61 characters, CLAUDE.md)
CJK = re.compile(r"[㐀-鿿豈-﫿]")
MISS_LABELS = ("ranking confusion", "bad description", "retrieval miss")


def description_flags(card: dict[str, Any]) -> list[str]:
    desc = (card.get("description") or "").strip()
    flags = []
    if not desc:
        flags.append("empty_description")
    elif len(desc) < SHORT_CHARS:
        flags.append("short_description")
    if CJK.search(desc):
        flags.append("non_english_description")
    if any(not (p.get("description") or "").strip() for p in card["params"]):
        flags.append("undocumented_params")
    return flags


def tool_health(cards: list[dict[str, Any]], gold: dict[str, dict[str, Any]], labels: list[dict[str, Any]],
                whatif: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per tool. `gold_caps` = capabilities that accept the tool; `missed` = retrieval-only rows (any of
    the 6 strategies) where such a capability was not found; `whatif` = best measured rewrite effect (a move
    into a top-10 first, then the largest rank gain), if any."""
    caps_of: dict[str, set[str]] = defaultdict(set)
    for task in gold.values():
        for cap in task["capabilities"]:
            for t in cap["acceptable"]:
                caps_of[t].add(cap["cap_id"])
    missed: dict[str, int] = defaultdict(int)
    for r in labels:
        if r["scope"] == "retrieval" and r["label"] in MISS_LABELS:
            cap_id = r["cap_id"]
            for t, caps in caps_of.items():
                if cap_id in caps:
                    missed[t] += 1
    best: dict[str, dict[str, Any]] = {}
    for w in whatif:
        gain = max(w["bm25"][0] - w["bm25"][1], w["semantic"][0] - w["semantic"][1])
        old = best.get(w["tool_id"])
        if old is None or (bool(w["moved_into_top10"]), gain) > (bool(old["moved_into_top10"]), old["gain"]):
            best[w["tool_id"]] = {"gain": gain, "bm25": w["bm25"], "semantic": w["semantic"],
                                  "moved_into_top10": w["moved_into_top10"]}
    rows = []
    for c in cards:
        t = c["tool_id"]
        rows.append({"tool_id": t, "server_key": c["server_key"], "description_chars": len(c["description"] or ""),
                     "flags": description_flags(c), "gold_caps": len(caps_of.get(t, ())),
                     "missed": missed.get(t, 0), "whatif": best.get(t)})
    return sorted(rows, key=lambda r: (-r["missed"], -len(r["flags"]), r["tool_id"]))

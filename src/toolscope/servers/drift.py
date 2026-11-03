"""Compare the benchmark catalog's tool schemas with the live servers' schemas (version drift).

breaking : tool missing live, a catalog parameter gone, or a new required parameter
additive : only new optional parameters (catalog-based calls still valid)
unchecked: live schema has no properties (server publishes an empty schema; calls not validated by it)
same     : identical parameter names and required sets

Run:  PYTHONPATH=src .venv/bin/python -m toolscope.servers.drift data/raw/tool_check/tools.json
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

Key = tuple[str, str]


def compare_tool(catalog: dict[str, Any], live: dict[str, Any] | None) -> str:
    if live is None:
        return "breaking"
    cp, lp = set(catalog.get("properties") or {}), set(live.get("properties") or {})
    if cp and not lp:
        return "unchecked"
    cr, lr = set(catalog.get("required") or []), set(live.get("required") or [])
    if cp - lp or lr - cr:
        return "breaking"
    return "same" if cp == lp and cr == lr else "additive"


def tool_drift(catalog: dict[Key, dict], live: dict[Key, dict]) -> dict[Key, str]:
    return {k: compare_tool(s, live.get(k)) for k, s in catalog.items()}


def load_schemas(path: Path) -> dict[Key, dict]:
    out = {}
    for server in json.loads(Path(path).read_text()):
        for key, block in server["tools"].items():
            for t in block["tools"]:
                out[(key, t["name"])] = t.get("inputSchema") or {}
    return out


def main() -> None:
    catalog = load_schemas(Path("third_party/LiveMCPBench/tools/LiveMCPTool/tools.json"))
    live = load_schemas(Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/tool_check/tools.json"))
    gold = {t for g in map(json.loads, Path("data/gold/gold.jsonl").read_text().splitlines())
            for c in g["capabilities"] for t in c["acceptable"]}
    drift = tool_drift(catalog, live)
    breaking = sorted(f"{s}/{t}" for (s, t), kind in drift.items() if kind == "breaking")
    gold_breaking = [t for t in breaking if t in gold]
    for t in breaking:
        print(f"  breaking{' GOLD' if t in gold else ''}: {t}")
    counts = Counter(drift.values())
    print(f"tools={len(drift)} same={counts['same']} additive={counts['additive']} unchecked={counts['unchecked']} "
          f"breaking={counts['breaking']} gold_breaking={len(gold_breaking)} "
          f"servers_breaking={len({t.split('/')[0] for t in breaking})}")


if __name__ == "__main__":
    main()

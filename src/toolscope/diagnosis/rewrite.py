"""Query-blind description rewrite (Phase 5 Step 3). The prompt is built from the tool card alone: server, tool
name, current description and parameters. It never receives a task or query (no leakage, CLAUDE.md).
The rubric names follow "MCP Tool Descriptions Are Smelly!" §4.1 (Purpose, Usage Guidelines, Limitations,
Parameter Explanation, Length & Completeness, Examples)."""

from typing import Any

PROMPT_VERSION = "rewrite-v1"
SYSTEM = (
    "You improve descriptions of MCP tools so that a search engine and an AI agent can find and use them. "
    "Rewrite the tool description in clear English, using only what the given information supports. Cover: "
    "Purpose (what the tool does and returns), Usage Guidelines (when to use it), Limitations (what it cannot do, "
    "if evident), Parameter Explanation (what each parameter means), Length & Completeness (2-5 sentences, under "
    "120 words), and one short usage example if helpful. If the description is not in English, translate its "
    "meaning. Do not invent features, data sources or parameters. Reply with the new description text only."
)


def card_prompt(card: dict[str, Any]) -> str:
    params = "\n".join(f"- {p['name']} ({p['type']}{', required' if p.get('required') else ''}): "
                       f"{p['description'] or '(no description)'}" for p in card["params"]) or "(none)"
    return (f"Server: {card['server_name']}\nServer description: {card.get('server_description') or '(none)'}\n"
            f"Tool name: {card['tool_name']}\nCurrent description: {card['description'] or '(empty)'}\n"
            f"Parameters:\n{params}")


def rewrite_messages(card: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": card_prompt(card)}]

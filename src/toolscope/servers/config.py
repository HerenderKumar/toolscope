"""Write ToolScope's own copy of the LiveMCPBench server config, with a per-server uv date pin.

The MCP stdio client passes each server only a whitelist of environment variables plus the
server's own `env`, so the pin goes into `env` as UV_EXCLUDE_NEWER (D-013, D-014).
third_party/ stays untouched; the output is used by the tool check and the Phase 4 gateway.

Run:  PYTHONPATH=src .venv/bin/python -m toolscope.servers.config
"""

import copy
import json
import os
from pathlib import Path
from typing import Any

LIVEMCPBENCH_DIR = Path(os.environ.get("TOOLSCOPE_LIVEMCPBENCH_DIR", "third_party/LiveMCPBench"))
PINS_PATH = Path(os.environ.get("TOOLSCOPE_SERVER_PINS", "config/server_pins.json"))
OUT_PATH = Path(os.environ.get("TOOLSCOPE_SERVER_CONFIG", "data/raw/server_config.json"))


def apply_pins(configs: list[dict[str, Any]], pins: dict[str, Any]) -> list[dict[str, Any]]:
    """Copy of `configs` where each server's env gets UV_EXCLUDE_NEWER (override, else default).
    An override pin of None leaves the server unpinned. An override `env` is merged in as-is;
    secrets go there only as ${VAR} placeholders, which the MCP client fills from its environment."""
    out = copy.deepcopy(configs)
    for server in out:
        for key, spec in server["config"]["mcpServers"].items():
            override = pins["overrides"].get(key)
            pin = override["pin"] if override else pins["default"]
            extra = dict((override or {}).get("env") or {})
            if pin:
                extra["UV_EXCLUDE_NEWER"] = pin
            # npm equivalent of the uv date pin (D-030): npx installs versions published before this date
            npm_before = (override or {}).get("npm_before", pins.get("npm_before"))
            if npm_before and spec["command"] == "npx":
                extra["npm_config_before"] = npm_before
            if extra:
                spec["env"] = {**(spec.get("env") or {}), **extra}
            if (override or {}).get("args"):  # exact package version, e.g. ["-y", "12306-mcp@0.3.9"] (D-030)
                spec["args"] = list(override["args"])
            prefix = (override or {}).get("command_prefix")
            if prefix:  # e.g. ["xvfb-run", "-a"]: run the server inside a wrapper
                spec["args"] = [*prefix[1:], spec["command"], *spec.get("args", [])]
                spec["command"] = prefix[0]
    return out


def main() -> None:
    configs = json.loads((LIVEMCPBENCH_DIR / "tools/LiveMCPTool/all_config.json").read_text())
    pins = json.loads(PINS_PATH.read_text())
    out = apply_pins(configs, pins)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    envs = [s.get("env", {}).get("UV_EXCLUDE_NEWER") for c in out for s in c["config"]["mcpServers"].values()]
    print(f"servers={len(out)} pinned_default={envs.count(pins['default'])} "
          f"pinned_other={sum(1 for e in envs if e and e != pins['default'])} "
          f"unpinned={envs.count(None)} out={OUT_PATH}")


if __name__ == "__main__":
    main()

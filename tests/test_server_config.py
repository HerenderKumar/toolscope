import copy

from toolscope.servers.config import apply_pins

CONFIGS = [
    {"name": "A", "config": {"mcpServers": {"a": {"command": "uvx", "args": ["a"]}}}},
    {"name": "B", "config": {"mcpServers": {"b": {"command": "uvx", "args": ["b"], "env": {"K": "1"}}}}},
    {"name": "C", "config": {"mcpServers": {"c": {"command": "npx", "args": ["c"]}}}},
]
PINS = {"default": "2026-07-28T13:44:00Z",
        "overrides": {"b": {"pin": "2025-08-03T00:00:00Z", "reason": "r"},
                      "c": {"pin": None, "reason": "r"}}}


def env_of(configs, key):
    for s in configs:
        if key in s["config"]["mcpServers"]:
            return s["config"]["mcpServers"][key].get("env")


def test_default_pin_applied():
    assert env_of(apply_pins(CONFIGS, PINS), "a") == {"UV_EXCLUDE_NEWER": "2026-07-28T13:44:00Z"}


def test_override_pin_keeps_existing_env():
    assert env_of(apply_pins(CONFIGS, PINS), "b") == {"K": "1", "UV_EXCLUDE_NEWER": "2025-08-03T00:00:00Z"}


def test_null_override_means_unpinned():
    assert env_of(apply_pins(CONFIGS, PINS), "c") is None


def test_input_not_mutated():
    before = copy.deepcopy(CONFIGS)
    apply_pins(CONFIGS, PINS)
    assert CONFIGS == before


def test_override_env_adds_placeholder_not_secret():
    pins = {"default": None, "overrides": {"c": {"pin": None, "env": {"API_KEY": "${API_KEY}"}, "reason": "r"}}}
    assert env_of(apply_pins(CONFIGS, pins), "c") == {"API_KEY": "${API_KEY}"}


def test_override_command_prefix_wraps_the_launch_command():
    pins = {"default": None, "overrides": {"c": {"pin": None, "command_prefix": ["xvfb-run", "-a"], "reason": "r"}}}
    spec = [s for s in apply_pins(CONFIGS, pins) if "c" in s["config"]["mcpServers"]][0]["config"]["mcpServers"]["c"]
    assert spec["command"] == "xvfb-run" and spec["args"] == ["-a", "npx", "c"]


def test_npm_before_applies_to_npx_servers_including_wrapped_ones():
    configs = [{"name": "N", "config": {"mcpServers": {"n": {"command": "npx", "args": ["-y", "pkg"]}}}},
               {"name": "W", "config": {"mcpServers": {"w": {"command": "npx", "args": ["-y", "pw"]}}}},
               {"name": "U", "config": {"mcpServers": {"u": {"command": "uvx", "args": ["x"]}}}}]
    pins = {"default": None, "npm_before": "2025-08-03",
            "overrides": {"w": {"pin": None, "command_prefix": ["xvfb-run", "-a"], "reason": "r"}}}
    out = {k: v for s in apply_pins(configs, pins) for k, v in s["config"]["mcpServers"].items()}
    assert out["n"]["env"] == {"npm_config_before": "2025-08-03"}
    assert out["w"]["env"] == {"npm_config_before": "2025-08-03"} and out["w"]["command"] == "xvfb-run"
    assert out["u"].get("env") is None


def test_npm_before_can_be_disabled_per_server():
    configs = [{"name": "N", "config": {"mcpServers": {"n": {"command": "npx", "args": ["-y", "pkg"]}}}}]
    pins = {"default": None, "npm_before": "2025-08-03", "overrides": {"n": {"pin": None, "npm_before": None, "reason": "r"}}}
    assert apply_pins(configs, pins)[0]["config"]["mcpServers"]["n"].get("env") is None


def test_override_args_replace_the_package_spec():
    pins = {"default": None, "npm_before": "2025-09-06",
            "overrides": {"c": {"pin": None, "npm_before": None, "args": ["-y", "c@0.3.9"], "reason": "r"}}}
    spec = [s for s in apply_pins(CONFIGS, pins) if "c" in s["config"]["mcpServers"]][0]["config"]["mcpServers"]["c"]
    assert spec["args"] == ["-y", "c@0.3.9"] and spec.get("env") is None

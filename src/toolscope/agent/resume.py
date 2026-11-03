"""Make long agent runs safe to interrupt: resume finished tasks, pause while the network is down."""

import json
import socket
import time
from pathlib import Path
from typing import Any, Callable

RUN_KEY_FIELDS = ("strategy", "backend", "model", "mode", "k", "temperature")
# records written before temperature was logged ran at temperature 0 (D-039)
KEY_DEFAULTS = {"temperature": 0.0}


def completed_task_ids(log_path: Path, key: dict[str, Any]) -> set[str]:
    """Tasks already finished cleanly for this run configuration. A task is redone if it errored
    or the network dropped while it ran (network_ok false). Records without the flag count as done."""
    if not log_path.exists():
        return set()
    done = set()
    for line in log_path.read_text().splitlines():
        rec = json.loads(line)
        if any(rec.get(f, KEY_DEFAULTS.get(f)) != key.get(f, KEY_DEFAULTS.get(f)) for f in RUN_KEY_FIELDS):
            continue
        if str(rec.get("stop_reason", "")).startswith("error") or rec.get("network_ok") is False:
            continue
        done.add(rec["task_id"])
    return done


# Host names, not IP literals: works on IPv4, IPv6-only and NAT64 networks. github.com and
# export.arxiv.org are IPv4-only services several tools need; if they are unreachable, tool
# results would fail for network reasons, so the run should wait instead.
NETWORK_PROBE_HOSTS = ("pypi.org", "github.com", "export.arxiv.org")


def network_ok(hosts: tuple[str, ...] = NETWORK_PROBE_HOSTS, port: int = 443, timeout_s: float = 6.0) -> bool:
    """True only if every probe host accepts a TCP connection on some resolved address."""
    for host in hosts:
        try:
            with socket.create_connection((host, port), timeout=timeout_s):
                pass
        except OSError:
            return False
    return True


def wait_for_network(check: Callable[[], bool] = network_ok, sleep: Callable[[float], None] = time.sleep,
                     interval_s: float = 30.0, on_wait: Callable[[float], None] | None = None) -> float:
    """Block until `check()` is true; returns the seconds waited."""
    waited = 0.0
    while not check():
        if on_wait:
            on_wait(waited)
        sleep(interval_s)
        waited += interval_s
    return waited

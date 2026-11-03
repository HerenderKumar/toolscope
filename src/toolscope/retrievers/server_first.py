"""Server-first: pick the top servers, then rank tools inside them (LiveMCPBench matcher.py, D-016)."""

import json
from pathlib import Path

import numpy as np

from toolscope.retrievers.base import Ranking
from toolscope.retrievers.dense import DenseRetriever


def server_first_rank(server_sims: dict[str, float], tool_sims: dict[str, float],
                      tool_server: dict[str, str], top_servers: int = 5) -> Ranking:
    """Tools on the top servers first, then the rest; within each group by
    final = (server_score * tool_sim) * max(server_score, tool_sim)."""
    kept = set(sorted(server_sims, key=lambda s: -server_sims[s])[:top_servers])

    def final(tid: str) -> float:
        s, t = server_sims[tool_server[tid]], tool_sims[tid]
        return (s * t) * max(s, t)

    def ordered(group: list[str]) -> list[str]:
        return sorted(group, key=lambda t: -final(t))

    first = ordered([t for t in tool_sims if tool_server[t] in kept])
    rest = ordered([t for t in tool_sims if tool_server[t] not in kept])
    ranking = [(t, final(t)) for t in first]
    floor = min((s for _, s in ranking), default=0.0)
    # keep scores strictly decreasing across the cut so ranks survive any score-based sort
    ranking += [(t, floor - 1.0 - i * 1e-3) for i, t in enumerate(rest)]
    return ranking


class ServerFirstRetriever:
    name = "server_first"

    def __init__(self, tool_dense: DenseRetriever, tool_index_dir: Path, server_index_dir: Path,
                 tool_server: dict[str, str], top_servers: int = 5):
        self.dense = tool_dense
        self.tool_vectors = np.load(Path(tool_index_dir) / "vectors.npy")
        self.tool_ids = json.loads((Path(tool_index_dir) / "ids.json").read_text())
        self.server_vectors = np.load(Path(server_index_dir) / "vectors.npy")
        self.server_keys = json.loads((Path(server_index_dir) / "ids.json").read_text())
        self.tool_server, self.top_servers = tool_server, top_servers

    def retrieve(self, query: str, k: int) -> Ranking:
        q = self.dense.encode_query(query)[0]
        server_sims = dict(zip(self.server_keys, (self.server_vectors @ q).tolist()))
        tool_sims = dict(zip(self.tool_ids, (self.tool_vectors @ q).tolist()))
        return server_first_rank(server_sims, tool_sims, self.tool_server, self.top_servers)[:k]

"""Hybrid: Reciprocal Rank Fusion of BM25 and ToolRet, k = 60 (Cormack et al. 2009; ranx fusion)."""

from ranx import Run, fuse

from toolscope.retrievers.base import Ranking, Retriever


def rrf(rankings: list[Ranking], k: int = 60) -> Ranking:
    """score(t) = sum over lists of 1 / (k + rank of t in that list)."""
    runs = [Run({"q": {tid: score for tid, score in r}}) for r in rankings]
    fused = fuse(runs=runs, method="rrf", params={"k": k}).to_dict()["q"]
    return sorted(fused.items(), key=lambda x: -x[1])


class HybridRetriever:
    name = "hybrid"

    def __init__(self, bm25: Retriever, toolret: Retriever, n_tools: int, k: int = 60):
        self.bm25, self.toolret, self.n_tools, self.k = bm25, toolret, n_tools, k

    def retrieve(self, query: str, k: int) -> Ranking:
        lists = [self.bm25.retrieve(query, self.n_tools), self.toolret.retrieve(query, self.n_tools)]
        return rrf(lists, self.k)[:k]

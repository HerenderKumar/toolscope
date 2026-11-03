"""Build any of the 6 retrievers by name. Indexes are loaded read-only; models are loaded once."""

import os
from functools import lru_cache

from toolscope.retrievers.base import INDEX_ROOT, STRATEGIES, Retriever, dense_doc_text, load_cards

MODELS = {
    "semantic": os.environ.get("TOOLSCOPE_SEMANTIC_MODEL", "BAAI/bge-base-en-v1.5"),
    "toolret": os.environ.get("TOOLSCOPE_TOOLRET_MODEL", "mangopy/ToolRet-trained-e5-base-v2"),
    "rerank": os.environ.get("TOOLSCOPE_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"),
}
# Per-model pooling and query prefix (D-016 for semantic; D-017 for toolret: the configuration of
# ToolRet's eval.py, which produced the model's published results).
MODEL_CONFIG = {
    "semantic": {"pooling": "cls", "query_prefix": "Represent this sentence for searching relevant passages: "},
    "toolret": {"pooling": "mean", "query_prefix": "Instruct: \nquery: "},
}
RERANK_TOP_N = int(os.environ.get("TOOLSCOPE_RERANK_TOP_N", "30"))
TOP_SERVERS = int(os.environ.get("TOOLSCOPE_TOP_SERVERS", "5"))
RRF_K = int(os.environ.get("TOOLSCOPE_RRF_K", "60"))


@lru_cache(maxsize=None)
def encoder(model_key: str):
    from toolscope.retrievers.dense import HFEncoder
    return HFEncoder(MODELS[model_key], pooling=MODEL_CONFIG[model_key]["pooling"])


@lru_cache(maxsize=None)
def get_retriever(name: str) -> Retriever:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; expected one of {STRATEGIES}")
    if name == "bm25":
        from toolscope.retrievers.bm25 import BM25Retriever
        return BM25Retriever(INDEX_ROOT / "bm25")
    if name in ("semantic", "toolret"):
        from toolscope.retrievers.dense import DenseRetriever
        return DenseRetriever(name, INDEX_ROOT / name, encoder(name), MODEL_CONFIG[name]["query_prefix"])
    n_tools = len(load_cards())
    if name == "hybrid":
        from toolscope.retrievers.hybrid import HybridRetriever
        return HybridRetriever(get_retriever("bm25"), get_retriever("toolret"), n_tools, RRF_K)
    if name == "rerank":
        from toolscope.retrievers.rerank import RerankRetriever, cross_encoder_scorer
        docs = {c["tool_id"]: dense_doc_text(c) for c in load_cards()}
        return RerankRetriever(get_retriever("hybrid"), cross_encoder_scorer(MODELS["rerank"]), docs,
                               RERANK_TOP_N, n_tools)
    from toolscope.retrievers.server_first import ServerFirstRetriever
    tool_server = {c["tool_id"]: c["server_key"] for c in load_cards()}
    return ServerFirstRetriever(get_retriever("toolret"), INDEX_ROOT / "toolret",
                                INDEX_ROOT / "servers_toolret", tool_server, TOP_SERVERS)

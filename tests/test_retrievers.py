import numpy as np
import pytest

from toolscope.retrievers.base import STRATEGIES, dense_doc_text
from toolscope.retrievers.bm25 import BM25Retriever, build_bm25_index
from toolscope.retrievers.dense import DenseRetriever, QUERY_PREFIX, build_dense_index
from toolscope.retrievers.hybrid import rrf
from toolscope.retrievers.rerank import RerankRetriever
from toolscope.retrievers.server_first import server_first_rank

CARDS = [
    {"tool_id": "weather/get_forecast", "server_key": "weather", "tool_name": "get_forecast",
     "description": "Get the weather forecast for a city",
     "params": [{"name": "city", "type": "string", "description": "City name", "required": True}],
     "search_text": "Weather | get forecast | get_forecast | Get the weather forecast for a city | city: City name"},
    {"tool_id": "fs/write_file", "server_key": "fs", "tool_name": "write_file",
     "description": "Write text content to a file on disk",
     "params": [{"name": "path", "type": "string", "description": "", "required": True}],
     "search_text": "Filesystem | write file | write_file | Write text content to a file on disk | path"},
    {"tool_id": "stocks/get_price", "server_key": "stocks", "tool_name": "get_price",
     "description": "Current stock price for a ticker symbol",
     "params": [],
     "search_text": "Stocks | get price | get_price | Current stock price for a ticker symbol"},
]


def ids(ranking):
    return [t for t, _ in ranking]


def test_six_strategies_registered():
    assert STRATEGIES == ["bm25", "semantic", "toolret", "hybrid", "rerank", "server_first"]


def test_dense_doc_text_matches_toolret_training_format():
    assert dense_doc_text(CARDS[0]) == (
        "{'name': 'get_forecast', 'description': 'Get the weather forecast for a city', "
        "'parameters': {'city': {'description': 'City name', 'type': 'string'}}}")


def test_bm25_ranks_word_match_first_and_returns_all(tmp_path):
    build_bm25_index(CARDS, tmp_path)
    ranking = BM25Retriever(tmp_path).retrieve("what is the stock price of AAPL", k=3)
    assert ids(ranking)[0] == "stocks/get_price"
    assert sorted(ids(ranking)) == sorted(c["tool_id"] for c in CARDS)


def test_bm25_query_with_no_known_words_still_returns_k(tmp_path):
    build_bm25_index(CARDS, tmp_path)
    assert len(BM25Retriever(tmp_path).retrieve("zzzz qqqq", k=2)) == 2


class FakeEncoder:
    """Maps a text to a fixed unit vector by keyword, so tests need no model."""
    seen: list[str] = []

    def encode(self, texts):
        out = []
        for t in texts:
            self.seen.append(t)
            v = np.array([("weather" in t.lower()) + 0.01, ("file" in t.lower()) + 0.01,
                          ("stock" in t.lower()) + 0.01], dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def test_dense_retriever_uses_prefix_and_cosine_order(tmp_path):
    enc = FakeEncoder()
    build_dense_index(CARDS, enc, tmp_path)
    ranking = DenseRetriever("toolret", tmp_path, enc).retrieve("save notes to a file", k=3)
    assert ids(ranking)[0] == "fs/write_file"
    assert enc.seen[-1] == QUERY_PREFIX + "save notes to a file"


def test_rrf_matches_formula():
    fused = rrf([[("a", 9.0), ("b", 5.0)], [("b", 0.9), ("c", 0.1)]], k=60)
    assert ids(fused) == ["b", "a", "c"]
    assert dict(fused)["b"] == pytest.approx(1 / 62 + 1 / 61)


class FakeBase:
    name = "hybrid"

    def retrieve(self, query, k):
        return [("t1", 0.9), ("t2", 0.8), ("t3", 0.7), ("t4", 0.6)][:k]


def test_rerank_reorders_head_and_keeps_tail_below():
    scorer = lambda q, docs: [{"d1": 0.1, "d2": 5.0}[d] for d in docs]
    r = RerankRetriever(FakeBase(), scorer, {"t1": "d1", "t2": "d2", "t3": "d3", "t4": "d4"}, top_n=2)
    ranking = r.retrieve("q", k=4)
    assert ids(ranking) == ["t2", "t1", "t3", "t4"]
    scores = [s for _, s in ranking]
    assert scores == sorted(scores, reverse=True) and len(set(scores)) == 4


def test_server_first_keeps_top_servers_first_and_uses_livemcpbench_formula():
    tool_server = {"a/x": "a", "b/y": "b", "c/z": "c"}
    server_sims = {"a": 0.9, "b": 0.2, "c": 0.5}
    tool_sims = {"a/x": 0.5, "b/y": 0.99, "c/z": 0.6}
    ranking = server_first_rank(server_sims, tool_sims, tool_server, top_servers=2)
    assert ids(ranking) == ["a/x", "c/z", "b/y"]  # b/y has the best tool_sim but its server is cut
    assert dict(ranking)["a/x"] > dict(ranking)["c/z"] > dict(ranking)["b/y"]


def test_registry_rejects_unknown_strategy():
    from toolscope.retrievers.registry import get_retriever
    with pytest.raises(ValueError, match="unknown strategy"):
        get_retriever("magic")


def test_pool_cls_takes_first_token_mean_averages_real_tokens():
    import torch
    from toolscope.retrievers.dense import pool
    hidden = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]]])  # last token is padding
    mask = torch.tensor([[1, 1, 0]])
    assert pool(hidden, mask, "cls").tolist() == [[1.0, 0.0]]
    assert pool(hidden, mask, "mean").tolist() == [[2.0, 0.0]]


def test_dense_retriever_uses_its_own_query_prefix(tmp_path):
    enc = FakeEncoder()
    build_dense_index(CARDS, enc, tmp_path)
    DenseRetriever("toolret", tmp_path, enc, query_prefix="Instruct: \nquery: ").retrieve("a file", k=1)
    assert enc.seen[-1] == "Instruct: \nquery: a file"


def test_model_configs_follow_d017():
    from toolscope.retrievers.registry import MODEL_CONFIG
    assert MODEL_CONFIG["toolret"] == {"pooling": "mean", "query_prefix": "Instruct: \nquery: "}
    assert MODEL_CONFIG["semantic"] == {"pooling": "cls",
                                        "query_prefix": "Represent this sentence for searching relevant passages: "}

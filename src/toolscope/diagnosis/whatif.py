"""What-if description editor: the rank of a tool for some queries before and after replacing its description,
under BM25 and the dense `semantic` retriever. Everything happens on in-memory copies: the BM25 index is
rebuilt in memory from the catalog, and dense vectors are read from indexes/semantic/vectors.npy and copied,
so the saved indexes stay read-only (CLAUDE.md). The editor never sees how a description was written, and
the rewrite step (scripts/rewrite_probe.py) never sees the queries.
"""

from pathlib import Path
from typing import Any, Callable

import bm25s
import numpy as np

from toolscope.catalog.build_catalog import make_search_text
from toolscope.retrievers.base import INDEX_ROOT, dense_doc_text, load_cards

Encode = Callable[[list[str]], np.ndarray]


def _tokenize(texts: list[str]) -> list[list[str]]:  # same settings as retrievers/bm25.py (D-016)
    return bm25s.tokenize(texts, stopwords="en", return_ids=False, show_progress=False)


def _bm25_scores(texts: list[str], query: str) -> np.ndarray:
    model = bm25s.BM25()
    model.index(_tokenize(texts), show_progress=False)
    tokens = [t for t in _tokenize([query])[0] if t in model.vocab_dict]
    return model.get_scores(tokens) if tokens else np.zeros(len(texts))


def rank_of(scores: np.ndarray, i: int) -> int:
    """1-based rank of item i under stable descending order (ties keep catalog order, as the retrievers do)."""
    order = np.argsort(-scores, kind="stable")
    return int(np.where(order == i)[0][0]) + 1


class WhatIf:
    def __init__(self, cards: list[dict[str, Any]] | None = None, vectors: np.ndarray | None = None,
                 encode_doc: Encode | None = None, encode_query: Encode | None = None):
        self.cards = cards if cards is not None else load_cards()
        self.ids = [c["tool_id"] for c in self.cards]
        self.vectors = vectors if vectors is not None else np.load(Path(INDEX_ROOT) / "semantic" / "vectors.npy")
        if encode_doc is None or encode_query is None:
            from toolscope.retrievers.registry import MODEL_CONFIG, encoder
            enc = encoder("semantic")
            prefix = MODEL_CONFIG["semantic"]["query_prefix"]
            encode_doc = encode_doc or enc.encode
            encode_query = encode_query or (lambda qs: enc.encode([prefix + q for q in qs]))
        self.encode_doc, self.encode_query = encode_doc, encode_query

    def _edited(self, tool_id: str, description: str) -> tuple[int, dict[str, Any]]:
        i = self.ids.index(tool_id)
        card = dict(self.cards[i], description=description)
        card["search_text"] = make_search_text(card["server_name"], card["tool_name"], description, card["params"])
        return i, card

    def whatif(self, tool_id: str, new_description: str, queries: list[str]) -> list[dict[str, Any]]:
        """[{query, bm25: [old, new], semantic: [old, new]}] ranks of tool_id (1-based, full catalog)."""
        i, card = self._edited(tool_id, new_description)
        texts = [c["search_text"] for c in self.cards]
        new_texts = texts[:i] + [card["search_text"]] + texts[i + 1:]
        vectors = self.vectors.copy()
        vectors[i] = self.encode_doc([dense_doc_text(card)])[0]
        out = []
        for q, qv in zip(queries, self.encode_query(queries)):
            out.append({"query": q,
                        "bm25": [rank_of(_bm25_scores(texts, q), i), rank_of(_bm25_scores(new_texts, q), i)],
                        "semantic": [rank_of(self.vectors @ qv, i), rank_of(vectors @ qv, i)]})
        return out

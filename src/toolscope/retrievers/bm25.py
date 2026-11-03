"""BM25 over catalog search_text with bm25s, configured as ToolRet's BM25 baseline (D-016)."""

import json
from pathlib import Path
from typing import Any

import bm25s
import numpy as np

from toolscope.retrievers.base import Ranking


def _tokenize(texts: list[str]) -> list[list[str]]:
    return bm25s.tokenize(texts, stopwords="en", return_ids=False, show_progress=False)


def build_bm25_index(cards: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = bm25s.BM25()
    model.index(_tokenize([c["search_text"] for c in cards]), show_progress=False)
    model.save(str(out_dir))
    (out_dir / "ids.json").write_text(json.dumps([c["tool_id"] for c in cards]))


class BM25Retriever:
    name = "bm25"

    def __init__(self, index_dir: Path):
        self.model = bm25s.BM25.load(str(index_dir), mmap=True)
        self.ids = json.loads((Path(index_dir) / "ids.json").read_text())

    def retrieve(self, query: str, k: int) -> Ranking:
        k = min(k, len(self.ids))
        tokens = [t for t in _tokenize([query])[0] if t in self.model.vocab_dict]
        if not tokens:  # no known word: every tool scores 0, keep catalog order
            return [(tid, 0.0) for tid in self.ids[:k]]
        scores = self.model.get_scores(tokens)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in order]

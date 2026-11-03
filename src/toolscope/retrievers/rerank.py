"""Cross-encoder rerank of the hybrid top-n; the rest keeps hybrid order below it (D-016)."""

from typing import Callable

from toolscope.retrievers.base import Ranking, Retriever

Scorer = Callable[[str, list[str]], list[float]]


def cross_encoder_scorer(model_id: str) -> Scorer:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_id, max_length=512)
    return lambda query, docs: [float(s) for s in model.predict([(query, d) for d in docs])]


class RerankRetriever:
    name = "rerank"

    def __init__(self, base: Retriever, scorer: Scorer, doc_texts: dict[str, str], top_n: int = 30,
                 n_tools: int | None = None):
        self.base, self.scorer, self.doc_texts, self.top_n = base, scorer, doc_texts, top_n
        self.n_tools = n_tools or len(doc_texts)

    def retrieve(self, query: str, k: int) -> Ranking:
        full = self.base.retrieve(query, max(k, self.n_tools))
        head, tail = full[:self.top_n], full[self.top_n:]
        scores = self.scorer(query, [self.doc_texts[t] for t, _ in head])
        reranked = sorted(zip([t for t, _ in head], scores), key=lambda x: -x[1])
        floor = min(scores) if scores else 0.0
        tail_scored = [(t, floor - 1.0 - i * 1e-3) for i, (t, _) in enumerate(tail)]
        return (reranked + tail_scored)[:k]

"""Dense retrieval: pooled, L2-normalised embeddings in an exact FAISS inner-product index.

Pooling and query prefix are per model (D-016, D-017)."""

import json
from pathlib import Path
from typing import Any, Protocol

# torch must load before faiss: both ship their own libomp on macOS, and the other order
# segfaults inside torch (exit 139) when encoding. Reproduced 28 Oct 2026; this order fixes it.
import torch  # noqa: F401
import faiss
import numpy as np

from toolscope.retrievers.base import Ranking, dense_doc_text

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


def pool(hidden, mask, mode: str):
    """'cls': the first token's vector. 'mean': average over real (non-padding) tokens."""
    if mode == "cls":
        return hidden[:, 0]
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class HFEncoder:
    """Transformer encoder with the model's own pooling, then L2 norm."""

    def __init__(self, model_id: str, pooling: str = "cls", max_length: int = 512, batch_size: int = 16):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval()
        self.pooling, self.max_length, self.batch_size = pooling, max_length, batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt")
            with self.torch.no_grad():
                hidden = self.model(**batch).last_hidden_state
            vectors = pool(hidden, batch["attention_mask"], self.pooling)
            out.append(self.torch.nn.functional.normalize(vectors, dim=-1).numpy())
        return np.concatenate(out).astype(np.float32)


def write_flat_index(vectors: np.ndarray, ids: list[str], out_dir: Path,
                     meta: dict[str, Any] | None = None) -> None:
    """Exact inner-product FAISS index + raw vectors + ids. The only place faiss writes (see import note)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    np.save(out_dir / "vectors.npy", vectors)
    (out_dir / "ids.json").write_text(json.dumps(ids))
    (out_dir / "meta.json").write_text(json.dumps({"normalized": True, **(meta or {})}, indent=2))


def build_dense_index(cards: list[dict[str, Any]], encoder: Encoder, out_dir: Path,
                      meta: dict[str, Any] | None = None) -> None:
    vectors = encoder.encode([dense_doc_text(c) for c in cards])
    write_flat_index(vectors, [c["tool_id"] for c in cards], out_dir, meta)


class DenseRetriever:
    def __init__(self, name: str, index_dir: Path, encoder: Encoder, query_prefix: str = QUERY_PREFIX):
        self.name, self.encoder, self.query_prefix = name, encoder, query_prefix
        self.index = faiss.read_index(str(Path(index_dir) / "index.faiss"))
        self.ids = json.loads((Path(index_dir) / "ids.json").read_text())

    def encode_query(self, query: str) -> np.ndarray:
        return self.encoder.encode([self.query_prefix + query])

    def retrieve(self, query: str, k: int) -> Ranking:
        scores, idx = self.index.search(self.encode_query(query), min(k, len(self.ids)))
        return [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0])]

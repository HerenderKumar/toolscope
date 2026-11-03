"""Build every index in indexes/: bm25, semantic, toolret, servers_toolret (for server-first).

Run:  HF_HUB_OFFLINE=1 PYTHONPATH=src .venv/bin/python scripts/build_indexes.py
"""

import time

# dense first: it imports torch before faiss (macOS libomp clash, see dense.py)
from toolscope.retrievers.dense import build_dense_index, write_flat_index
from toolscope.log import get_logger
from toolscope.retrievers.base import INDEX_ROOT, load_cards, server_descriptions
from toolscope.retrievers.bm25 import build_bm25_index
from toolscope.retrievers.registry import MODEL_CONFIG, MODELS, encoder

log = get_logger("toolscope.indexes")


def build_server_index(enc, out_dir) -> int:
    """Server descriptions, raw (no prefix), same encoder as ToolRet tool vectors."""
    servers = server_descriptions()
    keys = sorted(servers)
    write_flat_index(enc.encode([servers[k] for k in keys]), keys, out_dir, {"model": MODELS["toolret"]})
    return len(keys)


def main() -> None:
    cards = load_cards()
    t0 = time.time()
    build_bm25_index(cards, INDEX_ROOT / "bm25")
    for name in ("semantic", "toolret"):
        start = time.time()
        build_dense_index(cards, encoder(name), INDEX_ROOT / name, {"model": MODELS[name], **MODEL_CONFIG[name]})
        log.info("dense index built", extra={"ctx": {"name": name, "seconds": round(time.time() - start, 1)}})
    n_servers = build_server_index(encoder("toolret"), INDEX_ROOT / "servers_toolret")
    print(f"tools={len(cards)} bm25=1 dense=2 servers={n_servers} seconds={time.time() - t0:.0f} out={INDEX_ROOT}")


if __name__ == "__main__":
    main()

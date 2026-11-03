"""Download the 3 Hugging Face models into the local HF cache so later phases run offline.

Run:  .venv/bin/python scripts/prefetch_models.py
"""

import os

from huggingface_hub import snapshot_download

MODELS = [
    os.environ.get("TOOLSCOPE_SEMANTIC_MODEL", "BAAI/bge-base-en-v1.5"),
    os.environ.get("TOOLSCOPE_TOOLRET_MODEL", "mangopy/ToolRet-trained-e5-base-v2"),
    os.environ.get("TOOLSCOPE_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"),
]
# Only PyTorch/safetensors weights + configs; skip ONNX/OpenVINO/TF copies.
IGNORE = ["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot", "tf_model*"]


def main() -> None:
    ok = 0
    for repo in MODELS:
        path = snapshot_download(repo, ignore_patterns=IGNORE)
        ok += 1
        print(f"{repo} -> {path}")
    print(f"models={ok}/{len(MODELS)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pré-télécharge les modèles locaux (bge-m3 + reranker) dans HF_HOME.

Exécuté une seule fois par le service `models-init` du docker-compose (le seul moment où
HuggingFace est joint). Au runtime, HF_OFFLINE=true interdit tout téléchargement.
"""

from __future__ import annotations

import os

from huggingface_hub import snapshot_download

MODELS = ["BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"]
IGNORE = ["*.onnx", "onnx/*", "*.h5", "*.msgpack", "*.ckpt", "*.png", "*.jpg", "imgs/*"]


def main() -> None:
    target = os.environ.get("HF_HOME", "/models")
    marker = os.path.join(target, ".ready")
    if os.path.exists(marker):
        print(f"Modèles déjà présents dans {target}, rien à faire.")
        return
    for repo in MODELS:
        print(f"Téléchargement de {repo} …", flush=True)
        snapshot_download(repo_id=repo, ignore_patterns=IGNORE)
    os.makedirs(target, exist_ok=True)
    with open(marker, "w") as fh:
        fh.write("ok")
    print("Modèles prêts.")


if __name__ == "__main__":
    main()

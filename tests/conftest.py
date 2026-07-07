"""Fixtures partagées. Les tests d'intégration `slow` chargent les modèles locaux."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from rag_builder.core.store import QdrantStore

ROOT = Path(__file__).resolve().parent.parent
# Cache des modèles locaux (bge-m3, reranker) pré-téléchargés.
os.environ.setdefault("HF_HOME", str(ROOT / "storage" / "models_cache"))


@pytest.fixture
def store(tmp_path) -> QdrantStore:
    """QdrantStore local isolé (répertoire temporaire), fermé en fin de test."""
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    s = QdrantStore(client, is_local=True)
    try:
        yield s
    finally:
        s.close()


def _models_available() -> bool:
    hub = Path(os.environ["HF_HOME"]) / "hub"
    return (hub / "models--BAAI--bge-m3").exists()


requires_models = pytest.mark.skipif(
    not _models_available(),
    reason="Modèles locaux bge-m3 absents du cache (téléchargement requis).",
)

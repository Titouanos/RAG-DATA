"""Embedder local BAAI/bge-m3 (dense 1024 + sparse lexical), via FlagEmbedding.

100 % local, CPU. Multilingue (FR/EN). Fournit dans un seul passage le vecteur dense et
les poids lexicaux (sparse) exploités par le stockage hybride Qdrant.

Le modèle est chargé paresseusement (au premier appel) : construire l'objet est peu
coûteux. Les poids sont lus depuis le cache local (`models_cache_dir`) ; en production
réseau filtré, `hf_offline=True` interdit tout téléchargement au runtime.
"""

from __future__ import annotations

import logging
import os
import threading

from rag_builder.core.embeddings.base import Embedder, Embedding
from rag_builder.core.models import SparseVector

logger = logging.getLogger(__name__)


class LocalBGEM3Embedder(Embedder):
    """Implémentation par défaut : bge-m3 en local."""

    model_id = "BAAI/bge-m3"
    dense_dim = 1024
    supports_sparse = True

    def __init__(
        self,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        offline: bool = False,
        batch_size: int = 12,
        max_length: int = 8192,
        query_max_length: int = 512,
        model_name: str = "BAAI/bge-m3",
    ):
        self.model_id = model_name
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._offline = offline
        self._batch_size = batch_size
        self._max_length = max_length  # chunks de documents (jusqu'à ~2000 tokens)
        self._query_max_length = query_max_length  # requêtes courtes → cap bas = plus rapide
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            # HF_HOME doit être positionné AVANT le premier import de
            # transformers/FlagEmbedding. On s'appuie sur le cache HF standard
            # (`$HF_HOME/hub`) — le même que le pré-téléchargement — plutôt que sur
            # `cache_dir` (layout divergent → re-téléchargement).
            if self._cache_dir:
                os.environ["HF_HOME"] = self._cache_dir
            if self._offline:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from FlagEmbedding import BGEM3FlagModel

            logger.info("Chargement du modèle d'embedding %s (CPU)…", self.model_id)
            self._model = BGEM3FlagModel(self.model_id, use_fp16=False)
        return self._model

    def warm_up(self) -> None:
        """Force le chargement du modèle (utile au démarrage d'un worker long-vivant)."""
        self._ensure_model()

    # ------------------------------------------------------------------

    def _encode(self, texts: list[str], max_length: int | None = None) -> list[Embedding]:
        model = self._ensure_model()
        out = model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=max_length or self._max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"]
        lexical = out["lexical_weights"]
        results: list[Embedding] = []
        for i in range(len(texts)):
            dense_vec = [float(x) for x in dense[i]]
            sparse = _to_sparse(lexical[i])
            results.append(Embedding(dense=dense_vec, sparse=sparse))
        return results

    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> Embedding:
        return self._encode([text], max_length=self._query_max_length)[0]


def _to_sparse(lexical_weights: dict) -> SparseVector:
    """Convertit les poids lexicaux bge-m3 ({token_id(str): poids}) en SparseVector."""
    indices: list[int] = []
    values: list[float] = []
    for token_id, weight in lexical_weights.items():
        w = float(weight)
        if w <= 0.0:
            continue
        indices.append(int(token_id))
        values.append(w)
    return SparseVector(indices=indices, values=values)

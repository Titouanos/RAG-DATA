"""Reranker local : cross-encoder BAAI/bge-reranker-v2-m3 (FlagReranker), CPU.

Remplace le reranker LLM du POC (1 appel LLM/requête). Spécification conservée :
- skip si `len(candidats) <= top_k` (tri par score de fusion existant) ;
- fallback sur le tri de fusion en cas d'erreur du modèle ;
- top `rerank_k` → top `top_k`.

Activable par collection. Le modèle est chargé paresseusement, lu depuis le cache local.
"""

from __future__ import annotations

import logging
import os
import threading

from rag_builder.core.models import RetrievedChunk

logger = logging.getLogger(__name__)


class LocalReranker:
    """Cross-encoder local pour réordonner les candidats de retrieval."""

    def __init__(
        self,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        offline: bool = False,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 1024,
    ):
        self.model_id = model_name
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._offline = offline
        self._max_length = max_length
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            if self._cache_dir:
                os.environ["HF_HOME"] = self._cache_dir
            if self._offline:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from FlagEmbedding import FlagReranker

            logger.info("Chargement du reranker %s (CPU)…", self.model_id)
            self._model = FlagReranker(self.model_id, use_fp16=False)
        return self._model

    def warm_up(self) -> None:
        self._ensure_model()

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]:
        """Réordonne les candidats et retourne les `top_k` meilleurs."""
        if not candidates:
            return []
        if len(candidates) <= top_k:
            # Rien à départager au-delà du tri de fusion déjà fourni par le store.
            return candidates[:top_k]

        try:
            model = self._ensure_model()
            pairs = [[query, c.text] for c in candidates]
            scores = model.compute_score(pairs, normalize=True, max_length=self._max_length)
            if not isinstance(scores, list):
                scores = [scores]
            for c, s in zip(candidates, scores, strict=True):
                c.rerank_score = float(s)
            ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
            top = ranked[:top_k]
            for c in top:
                c.score = c.rerank_score if c.rerank_score is not None else c.score
            return top
        except Exception as exc:  # noqa: BLE001 — on ne casse jamais une requête sur le rerank
            logger.warning("Rerank échoué, fallback tri de fusion : %s", exc)
            return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]

"""Reranking local (cross-encoder), deux backends CPU.

- **onnx (défaut)** : `jinaai/jina-reranker-v2-base-multilingual` via fastembed (ONNX
  runtime), rapide sur CPU, multilingue FR/EN → tient le budget < 800 ms.
- **flag (option qualité)** : `BAAI/bge-reranker-v2-m3` via FlagEmbedding (XLM-RoBERTa-large,
  568M) — meilleure qualité mais ~15 s pour 20 candidats sur CPU (cf. mesures Phase 1).

Le backend est déduit du nom de modèle. Spécification commune : skip si
`len(candidats) <= top_k`, fallback sur le tri de fusion en cas d'erreur.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Protocol

from rag_builder.core.models import RetrievedChunk

logger = logging.getLogger(__name__)

# Modèles servis par fastembed (ONNX). Les autres passent par FlagEmbedding.
_FASTEMBED_MODELS = {
    "jinaai/jina-reranker-v2-base-multilingual",
    "jinaai/jina-reranker-v1-turbo-en",
    "jinaai/jina-reranker-v1-tiny-en",
    "Xenova/ms-marco-MiniLM-L-6-v2",
    "Xenova/ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-base",
}


class Reranker(Protocol):
    model_id: str

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]: ...

    def warm_up(self) -> None: ...


def build_reranker(
    model_name: str,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    offline: bool = False,
) -> Reranker:
    """Instancie le reranker adapté au modèle (fastembed ONNX ou FlagEmbedding)."""
    if model_name in _FASTEMBED_MODELS:
        return FastEmbedReranker(model_name, cache_dir=cache_dir, offline=offline)
    return FlagReranker(model_name, cache_dir=cache_dir, offline=offline)


def _apply_scores(
    candidates: list[RetrievedChunk], scores: list[float], top_k: int
) -> list[RetrievedChunk]:
    for c, s in zip(candidates, scores, strict=True):
        c.rerank_score = float(s)
    ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
    top = ranked[:top_k]
    for c in top:
        if c.rerank_score is not None:
            c.score = c.rerank_score
    return top


class FastEmbedReranker:
    """Reranker ONNX (fastembed TextCrossEncoder) — défaut, rapide sur CPU."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        offline: bool = False,
        max_passage_chars: int = 512,
    ):
        self.model_id = model_name
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._offline = offline
        # Troncature des passages : borne la latence CPU (cf. mesures Phase 1). Le préfixe
        # de contexte [Doc > H1 > H2] et le début du chunk portent l'essentiel du signal.
        self._max_passage_chars = max_passage_chars
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            if self._offline:
                os.environ["HF_HUB_OFFLINE"] = "1"
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            logger.info("Chargement du reranker ONNX %s…", self.model_id)
            self._model = TextCrossEncoder(model_name=self.model_id, cache_dir=self._cache_dir)
        return self._model

    def warm_up(self) -> None:
        self._ensure_model()

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        if len(candidates) <= top_k:
            return candidates[:top_k]
        try:
            model = self._ensure_model()
            passages = [c.text[: self._max_passage_chars] for c in candidates]
            scores = list(model.rerank(query, passages))
            return _apply_scores(candidates, scores, top_k)
        except Exception as exc:  # noqa: BLE001 — ne jamais casser une requête sur le rerank
            logger.warning("Rerank ONNX échoué, fallback tri de fusion : %s", exc)
            return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]


class FlagReranker:
    """Reranker FlagEmbedding (bge-reranker-v2-m3) — option qualité, lent sur CPU."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        offline: bool = False,
        max_length: int = 512,
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
            from FlagEmbedding import FlagReranker as _FlagReranker

            logger.info("Chargement du reranker %s (CPU)…", self.model_id)
            self._model = _FlagReranker(self.model_id, use_fp16=False)
        return self._model

    def warm_up(self) -> None:
        self._ensure_model()

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        if len(candidates) <= top_k:
            return candidates[:top_k]
        try:
            model = self._ensure_model()
            pairs = [[query, c.text] for c in candidates]
            scores = model.compute_score(pairs, normalize=True, max_length=self._max_length)
            if not isinstance(scores, list):
                scores = [scores]
            return _apply_scores(candidates, scores, top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Rerank échoué, fallback tri de fusion : %s", exc)
            return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]

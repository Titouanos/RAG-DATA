"""Wrapper autour de l'API Gemini pour les embeddings.

Utilise gemini-embedding-001 avec task_type spécifique pour maximiser la
qualité du retrieval :
- RETRIEVAL_DOCUMENT pour les chunks indexés
- RETRIEVAL_QUERY pour les queries utilisateur

La réduction matriochka (output_dimensionality) permet de descendre de 3072 à
une dimension plus petite sans retrain : 1536 est un bon compromis
qualité/coût/vitesse.

Les appels respectent le rate limiter (important pour le free tier).
"""

from __future__ import annotations

import logging
from typing import List

from rag.rate_limit import RateLimiter, with_rate_limit_retry

logger = logging.getLogger(__name__)


class GeminiEmbedder:
    """Embedder basé sur gemini-embedding-001."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001",
                 output_dim: int = 1536, batch_size: int = 100,
                 rate_limiter: RateLimiter | None = None):
        from google import genai
        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._rate_limiter = rate_limiter or RateLimiter()

    # ------------------------------------------------------------------
    # Documents (chunks à indexer)
    # ------------------------------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embedde une liste de chunks pour indexation."""
        return self._embed_batched(texts, task_type="RETRIEVAL_DOCUMENT")

    # ------------------------------------------------------------------
    # Queries (questions utilisateur)
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> List[float]:
        """Embedde une query utilisateur."""
        results = self._embed_batched([text], task_type="RETRIEVAL_QUERY")
        return results[0]

    # ------------------------------------------------------------------
    # Implémentation
    # ------------------------------------------------------------------

    def _embed_batched(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Appelle l'API par batch en respectant le rate limiter."""
        from google.genai import types

        # Hard cap : Gemini accepte max ~2K tokens par chunk (8K chars). Au-delà,
        # on tronque pour éviter un 400 sur 1M tokens. C'est un filet de sécurité
        # — le chunker normal devrait produire des chunks bien plus petits.
        # NB : on tronque silencieusement plutôt que de planter, car le doc qui
        # contient un chunk géant est généralement utile pour le reste.
        MAX_CHARS_PER_CHUNK = 30_000
        sanitized = []
        truncated_count = 0
        for t in texts:
            if len(t) > MAX_CHARS_PER_CHUNK:
                sanitized.append(t[:MAX_CHARS_PER_CHUNK])
                truncated_count += 1
            else:
                sanitized.append(t)
        if truncated_count:
            logger.warning(
                "%d chunks ont été tronqués à %d caractères (limite Gemini)",
                truncated_count, MAX_CHARS_PER_CHUNK,
            )

        results: List[List[float]] = []
        for start in range(0, len(sanitized), self._batch_size):
            batch = sanitized[start:start + self._batch_size]

            def _call(b=batch):
                config = types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self._output_dim,
                )
                return self._client.models.embed_content(
                    model=self._model,
                    contents=b,
                    config=config,
                )

            response = with_rate_limit_retry(
                _call, limiter=self._rate_limiter, model=self._model,
            )

            for emb in response.embeddings:
                values = list(emb.values)
                # Normalisation L2 (importante quand output_dim < 3072)
                values = _l2_normalize(values)
                results.append(values)

        return results

    @property
    def dim(self) -> int:
        return self._output_dim


def _l2_normalize(vec: List[float]) -> List[float]:
    """Normalisation L2. Important quand on utilise la réduction matriochka."""
    import math
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-12:
        return vec
    return [x / norm for x in vec]
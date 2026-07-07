"""Embedder Gemini (option, dense uniquement) — porté du POC.

Utilise `gemini-embedding-001` avec `task_type` asymétrique (RETRIEVAL_DOCUMENT vs
RETRIEVAL_QUERY) et réduction matriochka + normalisation L2. Ne fournit PAS de vecteur
sparse (`supports_sparse=False`) : une collection sur cet embedder est dense-only.
"""

from __future__ import annotations

import logging
import math

from rag_builder.core.embeddings.base import Embedder, Embedding

logger = logging.getLogger(__name__)

_MAX_CHARS = 30_000  # filet de sécurité contre les chunks géants (tronque + warn)


class GeminiEmbedder(Embedder):
    """Embedder basé sur l'API Gemini (dense-only)."""

    supports_sparse = False

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        output_dim: int = 1536,
        batch_size: int = 100,
    ):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model_id = model
        self.dense_dim = output_dim
        self._batch_size = batch_size

    def _embed(self, texts: list[str], task_type: str) -> list[Embedding]:
        from google.genai import types

        sanitized = [t[:_MAX_CHARS] for t in texts]
        results: list[Embedding] = []
        for start in range(0, len(sanitized), self._batch_size):
            batch = sanitized[start : start + self._batch_size]
            resp = self._client.models.embed_content(
                model=self.model_id,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=self.dense_dim
                ),
            )
            for emb in resp.embeddings:
                results.append(Embedding(dense=_l2_normalize([float(x) for x in emb.values])))
        return results

    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            return []
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> Embedding:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-12:
        return vec
    return [x / norm for x in vec]

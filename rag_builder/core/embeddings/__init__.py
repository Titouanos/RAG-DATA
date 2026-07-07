"""Fabrique d'embedders selon la configuration d'une collection."""

from __future__ import annotations

from rag_builder.core.embeddings.base import Embedder, Embedding

__all__ = ["Embedder", "Embedding", "build_embedder"]


def build_embedder(kind: str, settings) -> Embedder:
    """Instancie l'embedder correspondant (`local_bge_m3` par défaut).

    :param kind: `local_bge_m3` | `gemini` | `mistral`.
    :param settings: instance `rag_builder.config.Settings`.
    """
    kind = (kind or "local_bge_m3").lower()
    if kind == "local_bge_m3":
        from rag_builder.core.embeddings.local_bge_m3 import LocalBGEM3Embedder

        return LocalBGEM3Embedder(
            cache_dir=settings.models_cache_dir,
            offline=settings.hf_offline,
            model_name=settings.dense_model,
        )
    if kind == "gemini":
        from rag_builder.core.embeddings.gemini import GeminiEmbedder

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY requis pour l'embedder 'gemini'.")
        return GeminiEmbedder(api_key=settings.gemini_api_key)
    raise ValueError(f"Embedder inconnu : {kind!r} (attendu : local_bge_m3 | gemini | mistral)")

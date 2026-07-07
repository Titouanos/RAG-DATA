"""Fabrique de providers LLM selon la configuration d'une collection."""

from __future__ import annotations

from rag_builder.core.llm.base import LLMProvider

__all__ = ["LLMProvider", "build_provider"]


def build_provider(provider: str, model: str, settings) -> LLMProvider:
    """Instancie le provider LLM demandé.

    Phase 1 : `gemini` implémenté (réutilise la clé existante). `mistral`, `anthropic`,
    `ollama` arrivent en Phase 4.
    """
    provider = (provider or "gemini").lower()
    if provider == "gemini":
        from rag_builder.core.llm.gemini import GeminiProvider

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY requis pour le provider 'gemini'.")
        return GeminiProvider(api_key=settings.gemini_api_key, model=model)
    raise NotImplementedError(
        f"Provider LLM '{provider}' pas encore implémenté (Phase 4 : mistral/anthropic/ollama)."
    )

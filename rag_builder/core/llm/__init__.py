"""Fabrique de providers LLM selon la configuration d'une collection."""

from __future__ import annotations

from rag_builder.core.llm.base import LLMProvider

__all__ = ["LLMProvider", "build_provider"]


def build_provider(provider: str, model: str, settings) -> LLMProvider:
    """Instancie le provider LLM demandé.

    `mistral` (défaut cible, EU), `anthropic` (option qualité), `gemini`, `ollama` (local).
    """
    provider = (provider or "mistral").lower()

    if provider == "mistral":
        from rag_builder.core.llm.mistral import MistralProvider

        if not settings.mistral_api_key:
            raise ValueError("MISTRAL_API_KEY requis pour le provider 'mistral'.")
        return MistralProvider(api_key=settings.mistral_api_key, model=model)

    if provider == "gemini":
        from rag_builder.core.llm.gemini import GeminiProvider

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY requis pour le provider 'gemini'.")
        return GeminiProvider(api_key=settings.gemini_api_key, model=model)

    if provider == "anthropic":
        from rag_builder.core.llm.anthropic import AnthropicProvider

        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY requis pour le provider 'anthropic'.")
        return AnthropicProvider(api_key=settings.anthropic_api_key, model=model)

    if provider == "ollama":
        from rag_builder.core.llm.ollama import OllamaProvider

        return OllamaProvider(base_url=settings.ollama_base_url, model=model)

    raise ValueError(f"Provider LLM inconnu : {provider!r} (mistral|gemini|anthropic|ollama)")

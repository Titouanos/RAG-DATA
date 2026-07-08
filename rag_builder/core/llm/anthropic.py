"""Provider Anthropic (option qualité) — streaming via le SDK anthropic.

Modèles conseillés : `claude-haiku-4-5` (rapide) ou `claude-sonnet-5` (qualité). L'ID exact
est configurable par collection / via l'environnement.
"""

from __future__ import annotations

from collections.abc import Iterator

from rag_builder.core.llm.base import LLMProvider
from rag_builder.core.llm.retry import stream_with_retries


class AnthropicProvider(LLMProvider):
    """Génération streamée via l'API Anthropic (Messages)."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5"):
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self.model = model

    def stream(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        def factory() -> Iterator[str]:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=max_tokens or 4096,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                yield from stream.text_stream

        yield from stream_with_retries(factory, is_retriable=_is_retriable)


def _is_retriable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in (429, 500, 502, 503, 504, 529):
        return True
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connection", "apiconnection", "overloaded"))

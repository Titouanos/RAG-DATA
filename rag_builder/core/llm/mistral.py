"""Provider Mistral (défaut cible, hébergé en EU) — streaming via le SDK mistralai."""

from __future__ import annotations

from collections.abc import Iterator

from rag_builder.core.llm.base import LLMProvider
from rag_builder.core.llm.retry import stream_with_retries


class MistralProvider(LLMProvider):
    """Génération streamée via l'API Mistral."""

    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        try:
            from mistralai import Mistral  # SDK 1.x (export top-level)
        except ImportError:
            from mistralai.client import Mistral  # SDK 2.x (sous-module client)

        self._client = Mistral(api_key=api_key)
        self.model = model

    def stream(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        def factory() -> Iterator[str]:
            events = self._client.chat.stream(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for event in events:
                delta = event.data.choices[0].delta.content
                if delta:
                    # Certaines versions renvoient une liste de chunks de contenu.
                    yield delta if isinstance(delta, str) else "".join(
                        getattr(c, "text", "") for c in delta
                    )

        yield from stream_with_retries(factory, is_retriable=_is_retriable)


def _is_retriable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connection", "transport"))

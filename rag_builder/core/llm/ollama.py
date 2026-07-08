"""Provider Ollama (100 % local, option corpus sensible) — streaming via HTTP /api/chat."""

from __future__ import annotations

import json
from collections.abc import Iterator

from rag_builder.core.llm.base import LLMProvider
from rag_builder.core.llm.retry import stream_with_retries


class OllamaProvider(LLMProvider):
    """Génération streamée via un serveur Ollama local."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1"):
        self._base = base_url.rstrip("/")
        self.model = model

    def stream(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        import httpx

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        def factory() -> Iterator[str]:
            with httpx.stream(
                "POST", f"{self._base}/api/chat", json=payload, timeout=120.0
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    tok = data.get("message", {}).get("content")
                    if tok:
                        yield tok

        yield from stream_with_retries(factory, is_retriable=_is_retriable)


def _is_retriable(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connect", "transport"))

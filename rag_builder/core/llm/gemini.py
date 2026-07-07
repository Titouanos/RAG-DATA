"""Provider LLM Gemini (streaming) — réutilise la clé existante (défaut Phase 1).

Mistral (`mistral-large-latest`) reste le provider par défaut cible ; il est câblé en
Phase 4 derrière la même interface `LLMProvider`.
"""

from __future__ import annotations

from collections.abc import Iterator

from rag_builder.core.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    """Génération streamée via l'API Gemini."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model = model

    def stream(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        for chunk in self._client.models.generate_content_stream(
            model=self.model, contents=prompt, config=config
        ):
            if chunk.text:
                yield chunk.text

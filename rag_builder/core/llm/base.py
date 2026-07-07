"""Interface `LLMProvider` : génération en streaming (obligatoire).

Toutes les implémentations (gemini, mistral, anthropic, ollama) exposent `stream()` qui
produit les tokens au fil de l'eau — c'est ce flux que l'API relaie en SSE. `generate()`
est un utilitaire non-streamé (concatène le flux) pour le CLI et les tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMProvider(ABC):
    """Contrat commun aux providers de génération."""

    #: Identifiant du modèle (ex. "mistral-large-latest", "gemini-2.5-flash").
    model: str

    @abstractmethod
    def stream(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Génère la réponse token par token."""

    def generate(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Génère la réponse complète (concatène le flux)."""
        return "".join(
            self.stream(system, prompt, temperature=temperature, max_tokens=max_tokens)
        )

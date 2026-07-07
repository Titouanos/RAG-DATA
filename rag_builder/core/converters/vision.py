"""Interface `VisionDescriber` pour la description d'images (multimodal, optionnel).

En Phase 1 la vision est désactivée par défaut (`vision_enabled=False`) : les converters
reçoivent `vision_describer=None` et n'extraient pas les descriptions d'images. Une
implémentation Gemini est fournie (réutilise la clé existante) ; d'autres providers
(pixtral, claude) viendront ultérieurement derrière la même interface.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class VisionDescriber(Protocol):
    """Décrit une image en texte (français, factuel)."""

    def describe(self, image_bytes: bytes, mime: str) -> str: ...


VISION_PROMPT = (
    "Décris factuellement le contenu de cette image en français, en 300 mots maximum. "
    "Retranscris le texte lisible (boutons, menus, libellés), les schémas et les étapes. "
    "N'invente rien ; si l'image est illisible ou décorative, réponds par une courte "
    "mention neutre."
)


class GeminiVisionDescriber:
    """Description d'images via Gemini (optionnel, extra `gemini`)."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite"):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def describe(self, image_bytes: bytes, mime: str) -> str:
        from google.genai import types

        try:
            resp = self._client.models.generate_content(
                model=self._model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime),
                    VISION_PROMPT,
                ],
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=800),
            )
            return (resp.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Description Vision échouée : %s", exc)
            return ""

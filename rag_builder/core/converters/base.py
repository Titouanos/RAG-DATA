"""Base des converters : protocole, registre, hachage, `doc_id` path-indépendant.

Un converter transforme une source (fichier) en `ConvertedDoc` (markdown unifié). Le
`doc_id` est dérivé du **nom de la source** (pas du chemin absolu — corrige le risque R4
de l'état des lieux), donc stable au déplacement du fichier. Le `content_hash` (sha256 du
markdown) porte la détection de modification pour l'ingestion incrémentale.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from rag_builder.core.models import ConvertedDoc

logger = logging.getLogger(__name__)


def make_doc_id(source_name: str) -> str:
    """ID de document stable dérivé du nom de la source (indépendant du chemin)."""
    digest = hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:16]
    return f"doc_{digest}"


def hash_content(text: str) -> str:
    """sha256 complet du markdown converti (détection de modification)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@runtime_checkable
class Converter(Protocol):
    """Interface d'un converter (duck-typée)."""

    def can_handle(self, source: Path) -> bool: ...

    def convert(self, source: Path) -> ConvertedDoc | None: ...


class ConverterRegistry:
    """Route chaque source vers le premier converter capable de la traiter."""

    def __init__(self, converters: list[Converter]):
        self._converters = converters

    def convert(self, source: Path) -> ConvertedDoc | None:
        for conv in self._converters:
            try:
                if conv.can_handle(source):
                    return conv.convert(source)
            except Exception:
                logger.exception("Converter %s a échoué sur %s", type(conv).__name__, source)
                raise
        logger.debug("Aucun converter pour %s", source)
        return None


def iter_sources(data_dir: Path) -> Iterator[Path]:
    """Scan récursif trié de `data_dir` (fichiers uniquement, hors dotfiles).

    Contrairement au POC, plus de traitement spécial des `.txt` d'URLs YouTube
    (converters vidéo/YouTube retirés de la v1).
    """
    if not data_dir.exists():
        return
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            yield path

"""Stockage des images extraites des documents pour le RAG multimodal.

Format de référence dans le markdown :
    ![desc](rag-image://<doc_id>/<img_hash>.<ext>)

Le scheme `rag-image://` est interne et sera réécrit en URL HTTP par le
serveur MCP au moment de servir la réponse à OpenWebUI.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


INTERNAL_SCHEME = "rag-image://"


@dataclass
class StoredImage:
    doc_id: str
    image_id: str
    extension: str
    relative_path: str
    absolute_path: Path

    @property
    def reference(self) -> str:
        return f"{INTERNAL_SCHEME}{self.relative_path}"


class ImageStore:
    """Stockage des images extraites des documents.

    Toutes les images vivent sous storage/images/<doc_id>/<img_hash>.<ext>.
    Le hash bytes assure que la même image (même contenu) n'est stockée
    qu'une fois par doc, et garantit l'idempotence des ré-ingestions.
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save(self, doc_id: str, image_bytes: bytes,
             extension: str = ".png") -> StoredImage:
        """Sauvegarde une image et retourne sa référence (idempotent)."""
        ext = extension.lower()
        if not ext.startswith("."):
            ext = "." + ext
        if ext not in self.SUPPORTED_EXTENSIONS:
            ext = ".png"

        image_id = hashlib.sha256(image_bytes).hexdigest()[:20]
        doc_dir = self.root_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{image_id}{ext}"
        absolute = doc_dir / filename
        if not absolute.exists():
            absolute.write_bytes(image_bytes)

        relative = f"{doc_id}/{filename}"
        return StoredImage(
            doc_id=doc_id, image_id=image_id, extension=ext,
            relative_path=relative, absolute_path=absolute,
        )

    def remove_doc(self, doc_id: str) -> int:
        """Supprime toutes les images d'un doc."""
        doc_dir = self.root_dir / doc_id
        if not doc_dir.exists():
            return 0
        count = 0
        for f in doc_dir.iterdir():
            try:
                f.unlink()
                count += 1
            except OSError:
                pass
        try:
            doc_dir.rmdir()
        except OSError:
            pass
        return count

    def resolve(self, relative_path: str) -> Path | None:
        """Résout une référence interne en chemin absolu (avec sécurité anti path-traversal)."""
        relative_path = relative_path.lstrip("/\\")
        candidate = (self.root_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.root_dir.resolve())
        except ValueError:
            logger.warning("Tentative d'accès hors images_root: %s", relative_path)
            return None
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

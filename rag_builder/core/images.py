"""Stockage content-addressed des images extraites des documents (multi-collections).

Référence interne insérée dans le markdown :
    ![desc](rag-image://<collection>/<doc_id>/<sha256[:20]>.<ext>)

Le scheme `rag-image://` est réécrit en URL HTTP par l'API (ou le serveur MCP) au moment
de servir la réponse. Le stockage est idempotent (nom = hash du contenu).
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

INTERNAL_SCHEME = "rag-image://"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

# Identifiants sûrs pour le filesystem (collection, doc_id) : pas de séparateur ni de "..".
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _check_safe_id(value: str, kind: str) -> None:
    if not value or value in {".", ".."} or not _SAFE_ID_RE.match(value):
        raise ValueError(f"{kind} invalide pour le stockage d'images : {value!r}")


@dataclass
class StoredImage:
    collection: str
    doc_id: str
    image_id: str
    extension: str
    relative_path: str  # <collection>/<doc_id>/<hash>.<ext>
    absolute_path: Path

    @property
    def reference(self) -> str:
        return f"{INTERNAL_SCHEME}{self.relative_path}"


class ImageStore:
    """Stockage des images sous `root_dir/<collection>/<doc_id>/<hash>.<ext>`."""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self, collection: str, doc_id: str, image_bytes: bytes, extension: str = ".png"
    ) -> StoredImage:
        """Sauvegarde une image et retourne sa référence (idempotent)."""
        _check_safe_id(collection, "collection")
        _check_safe_id(doc_id, "doc_id")

        ext = extension.lower()
        if not ext.startswith("."):
            ext = "." + ext
        if ext not in SUPPORTED_EXTENSIONS:
            ext = ".png"

        image_id = hashlib.sha256(image_bytes).hexdigest()[:20]
        doc_dir = self.root_dir / collection / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{image_id}{ext}"
        absolute = doc_dir / filename
        if not absolute.exists():
            absolute.write_bytes(image_bytes)

        relative = f"{collection}/{doc_id}/{filename}"
        return StoredImage(
            collection=collection,
            doc_id=doc_id,
            image_id=image_id,
            extension=ext,
            relative_path=relative,
            absolute_path=absolute,
        )

    def remove_doc(self, collection: str, doc_id: str) -> int:
        """Supprime toutes les images d'un document. Retourne le nombre supprimé."""
        _check_safe_id(collection, "collection")
        _check_safe_id(doc_id, "doc_id")
        doc_dir = self.root_dir / collection / doc_id
        if not doc_dir.exists():
            return 0
        count = 0
        for f in doc_dir.iterdir():
            try:
                f.unlink()
                count += 1
            except OSError as exc:
                logger.warning("Suppression image échouée %s : %s", f, exc)
        with contextlib.suppress(OSError):
            doc_dir.rmdir()
        return count

    def remove_collection(self, collection: str) -> None:
        """Supprime toutes les images d'une collection."""
        _check_safe_id(collection, "collection")
        coll_dir = self.root_dir / collection
        if not coll_dir.exists():
            return
        import shutil

        shutil.rmtree(coll_dir, ignore_errors=True)

    def resolve(self, relative_path: str) -> Path | None:
        """Résout une référence relative en chemin absolu (anti path-traversal)."""
        relative_path = relative_path.lstrip("/\\")
        candidate = (self.root_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.root_dir.resolve())
        except ValueError:
            logger.warning("Accès hors racine images refusé : %s", relative_path)
            return None
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

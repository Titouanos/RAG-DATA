"""Extraction d'archives ZIP en fichiers ingérables (un document par fichier supporté).

Utilisé par l'upload API : déposer un `.zip` (ex. export de doc GLPI en HTML) crée un job
d'ingestion par fichier supporté qu'il contient. Le `source_name` de chaque document est
son **chemin relatif dans l'archive** (ex. `glpi/faq/reset.html`) — visible dans les
citations et stable pour la déduplication/suppression.

Garde-fous : protection zip-slip (chemins traversants), fichiers cachés et `__MACOSX`
ignorés, plafond de taille par fichier et de nombre d'entrées (zip bomb).
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from rag_builder.core.converters import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

ZIP_EXTENSIONS = {".zip"}
DEFAULT_MAX_ENTRIES = 2000


@dataclass
class ExtractedFile:
    """Fichier extrait d'une archive, prêt à être ingéré."""

    path: Path  # chemin extrait sur disque
    source_name: str  # chemin relatif dans l'archive (identité du document)
    size_bytes: int


@dataclass
class ZipReport:
    """Résultat d'une extraction : fichiers retenus + comptage des ignorés."""

    files: list[ExtractedFile] = field(default_factory=list)
    skipped_unsupported: int = 0
    skipped_too_big: int = 0
    skipped_unsafe: int = 0


def is_zip(filename: str) -> bool:
    return Path(filename).suffix.lower() in ZIP_EXTENSIONS


def expand_zip(
    zip_path: Path,
    dest_dir: Path,
    *,
    max_file_bytes: int,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> ZipReport:
    """Extrait les fichiers supportés de `zip_path` sous `dest_dir`.

    Retourne un `ZipReport` ; ne lève pas pour les entrées individuelles problématiques
    (elles sont comptées), mais propage `zipfile.BadZipFile` si l'archive est illisible.
    """
    report = ZipReport()
    dest_root = dest_dir.resolve()

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            raw_name = info.filename.replace("\\", "/")
            base_name = Path(raw_name).name
            # Métadonnées macOS et fichiers cachés.
            if "__MACOSX" in raw_name or base_name.startswith("."):
                continue
            if Path(raw_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                report.skipped_unsupported += 1
                continue
            if info.file_size > max_file_bytes:
                logger.warning("ZIP : %s dépasse la taille max, ignoré.", raw_name)
                report.skipped_too_big += 1
                continue
            if len(report.files) >= max_entries:
                logger.warning("ZIP : plafond de %d fichiers atteint, le reste est ignoré.",
                               max_entries)
                break

            # Chemin relatif propre (identité du document) + protection zip-slip.
            source_name = raw_name.lstrip("/")
            target = (dest_dir / source_name).resolve()
            if not target.is_relative_to(dest_root):
                logger.warning("ZIP : chemin traversant refusé : %s", raw_name)
                report.skipped_unsafe += 1
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            report.files.append(
                ExtractedFile(path=target, source_name=source_name, size_bytes=info.file_size)
            )

    return report

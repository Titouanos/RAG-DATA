"""Registre des converters concrets et fabrique du registre par défaut.

L'ordre des converters compte : les converters spécifiques passent avant markitdown, qui
sert de fallback générique. En particulier ``office_legacy`` précède ``markitdown`` alors
que markitdown gère les OOXML — mais les deux traitent des extensions disjointes (legacy
binaire vs OOXML), l'ordre garantit surtout que markitdown reste le dernier recours.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from rag_builder.core.converters.base import (
    Converter,
    ConverterRegistry,
    hash_content,
    iter_sources,
    make_doc_id,
)
from rag_builder.core.converters.markitdown_conv import MarkitdownConverter
from rag_builder.core.converters.mindmap import MindManagerConverter
from rag_builder.core.converters.office_legacy import LibreOfficeConverter
from rag_builder.core.converters.pdf import PdfConverter

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "Converter",
    "ConverterRegistry",
    "LibreOfficeConverter",
    "MarkitdownConverter",
    "MindManagerConverter",
    "PdfConverter",
    "build_default_registry",
    "hash_content",
    "iter_sources",
    "make_doc_id",
]

# Toutes les extensions ingérables par le registre par défaut (pour filtrer les archives,
# valider les uploads, etc.).
SUPPORTED_EXTENSIONS: set[str] = (
    {".pdf", ".mmap"}
    | {".doc", ".dot", ".xls", ".xlt", ".ppt", ".pot"}  # legacy via LibreOffice
    | MarkitdownConverter.SUPPORTED_EXTS
)


def build_default_registry(
    collection: str,
    *,
    image_store=None,
    vision_describer=None,
    vision_cache_dir: Path | None = None,
    ocr_enabled: bool = False,
    ocr_languages: str = "fra+eng",
    image_roots: list[Path] | None = None,
) -> ConverterRegistry:
    """Construit le registre par défaut.

    Ordre : mindmap, pdf, office_legacy, markitdown (fallback). Pour PDF/mindmap,
    l'extraction d'images exige ``image_store`` **et** ``vision_describer`` ; pour le
    HTML/MD, ``image_store`` suffit (les descriptions Vision restent optionnelles).
    L'OCR PDF (``ocr_enabled``) nécessite ocrmypdf + Tesseract. ``image_roots`` borne la
    résolution des images relatives des pages HTML (anti-traversée).
    """
    # Cache LibreOffice : à côté du cache Vision si fourni, sinon dans le temp système.
    if vision_cache_dir is not None:
        office_cache_dir = Path(vision_cache_dir).parent / "office_cache"
    else:
        office_cache_dir = Path(tempfile.gettempdir()) / "rag_builder_office_cache"

    markitdown = MarkitdownConverter(
        collection,
        image_store=image_store,
        vision_describer=vision_describer,
        vision_cache_dir=vision_cache_dir,
        image_roots=image_roots,
    )

    converters: list[Converter] = [
        MindManagerConverter(
            collection,
            vision_describer=vision_describer,
            image_store=image_store,
            vision_cache_dir=vision_cache_dir,
        ),
        PdfConverter(
            collection,
            image_store=image_store,
            vision_describer=vision_describer,
            vision_cache_dir=vision_cache_dir,
            ocr_enabled=ocr_enabled,
            ocr_languages=ocr_languages,
        ),
        LibreOfficeConverter(office_cache_dir, markitdown),
        markitdown,
    ]
    return ConverterRegistry(converters)

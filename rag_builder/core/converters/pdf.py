"""Converter PDF : extraction du texte page par page via PyMuPDF, images optionnelles.

Le texte de chaque page devient une section markdown ``## Page N``. Le titre est déduit
des métadonnées PDF si elles paraissent crédibles, sinon du nom de fichier. Les pages sans
couche texte sont repérées (``metadata["empty_text_pages"]``) et, si plus de 30 % des pages
sont vides, ``metadata["scanned_suspect"]`` est posé pour suggérer un OCR ultérieur
(l'OCR lui-même relève de la Phase 4 et n'est pas implémenté ici).

L'extraction des images embarquées ne s'active que si un ``image_store`` **et** un
``vision_describer`` sont fournis ; sinon le converter produit du texte seul.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from rag_builder.core.converters._image_md import (
    build_image_block,
    is_decorative_image,
    mime_for_ext,
)
from rag_builder.core.converters.base import hash_content, make_doc_id
from rag_builder.core.models import ConvertedDoc

logger = logging.getLogger(__name__)

# Fraction de pages sans texte au-delà de laquelle le PDF est suspecté scanné.
_SCANNED_SUSPECT_RATIO = 0.30

_BAD_META_TITLES = {
    "ethan frome",
    "document",
    "document1",
    "untitled",
    "microsoft word",
    "presentation1",
    "book1",
    "classeur1",
}


class PdfConverter:
    """Converter dédié aux PDF (texte page par page, images embarquées optionnelles)."""

    SUPPORTED_EXTS = {".pdf"}

    # Filtres pour ignorer les images décoratives (repris du POC).
    MIN_IMAGE_WIDTH = 80
    MIN_IMAGE_HEIGHT = 60
    MIN_IMAGE_BYTES = 2000

    def __init__(
        self,
        collection: str,
        image_store=None,
        vision_describer=None,
        vision_cache_dir: Path | None = None,
        ocr_enabled: bool = False,
        ocr_languages: str = "fra+eng",
    ):
        self.collection = collection
        self.image_store = image_store
        self.vision_describer = vision_describer
        self.vision_cache_dir = vision_cache_dir
        self.ocr_enabled = ocr_enabled
        self.ocr_languages = ocr_languages
        # L'extraction d'images exige à la fois le stockage et la vision.
        self.enable_images = image_store is not None and vision_describer is not None

    def can_handle(self, source: Path) -> bool:
        return source.suffix.lower() in self.SUPPORTED_EXTS

    def convert(self, source: Path) -> ConvertedDoc | None:
        try:
            import fitz  # PyMuPDF (import paresseux)
        except ImportError:
            logger.error("PyMuPDF n'est pas installé (pip install pymupdf) : PDF ignoré.")
            return None

        path = Path(source)
        doc_id = make_doc_id(path.name)

        try:
            pdf = fitz.open(str(path))
        except Exception as exc:  # noqa: BLE001
            logger.error("Impossible d'ouvrir %s : %s", path.name, exc)
            return None

        title = path.stem
        meta = pdf.metadata or {}
        meta_title = (meta.get("title") or "").strip()
        if meta_title and self._title_looks_legit(meta_title, path.stem):
            title = meta_title

        try:
            markdown_parts, empty_text_pages, page_count = self._extract_all(pdf, doc_id, title)
        finally:
            pdf.close()

        ocr_applied = False
        # OCR optionnel : si activé et des pages sans texte sont détectées, on ré-océrise.
        if self.ocr_enabled and empty_text_pages:
            ocr_path = self._run_ocr(path)
            if ocr_path is not None:
                try:
                    pdf2 = fitz.open(str(ocr_path))
                    try:
                        markdown_parts, empty_text_pages, page_count = self._extract_all(
                            pdf2, doc_id, title
                        )
                        ocr_applied = True
                    finally:
                        pdf2.close()
                finally:
                    ocr_path.unlink(missing_ok=True)

        markdown = "\n".join(markdown_parts).strip()
        if not markdown:
            logger.warning("PDF vide après conversion : %s", path.name)
            return None

        metadata: dict = {
            "filename": path.name,
            "filepath": str(path),
            "pages": page_count,
        }
        if ocr_applied:
            metadata["ocr_applied"] = True
        if empty_text_pages:
            metadata["empty_text_pages"] = empty_text_pages
            if page_count and len(empty_text_pages) / page_count > _SCANNED_SUSPECT_RATIO:
                metadata["scanned_suspect"] = True

        return ConvertedDoc(
            doc_id=doc_id,
            source_name=path.name,
            title=title,
            markdown=markdown,
            content_hash=hash_content(markdown),
            doc_type="pdf",
            metadata=metadata,
        )

    def _extract_all(self, pdf, doc_id: str, title: str) -> tuple[list[str], list[int], int | None]:
        """Extrait le markdown page par page. Retourne (parts, pages_sans_texte, nb_pages)."""
        markdown_parts: list[str] = [f"# {title}", ""]
        page_count = getattr(pdf, "page_count", None)
        empty_text_pages: list[int] = []
        for page_num, page in enumerate(pdf, start=1):
            has_text, page_md = self._convert_page(page, page_num, doc_id)
            if not has_text:
                empty_text_pages.append(page_num)
            if page_md.strip():
                markdown_parts.append(page_md)
                markdown_parts.append("")
        return markdown_parts, empty_text_pages, page_count

    def _run_ocr(self, source: Path) -> Path | None:
        """Ajoute une couche texte via ocrmypdf (Tesseract). None si indisponible/échec."""
        if not shutil.which("ocrmypdf"):
            logger.warning(
                "OCR demandé mais 'ocrmypdf' est absent du PATH — %s laissé tel quel.",
                source.name,
            )
            return None
        out = Path(tempfile.mkdtemp(prefix="ragb_ocr_")) / f"ocr_{source.name}"
        try:
            subprocess.run(
                ["ocrmypdf", "--skip-text", "--language", self.ocr_languages,
                 str(source), str(out)],
                check=True,
                capture_output=True,
                timeout=600,
            )
            logger.info("OCR appliqué à %s (%s)", source.name, self.ocr_languages)
            return out
        except Exception as exc:  # noqa: BLE001 — l'OCR ne doit jamais casser l'ingestion
            logger.warning("OCR échoué sur %s : %s", source.name, exc)
            out.unlink(missing_ok=True)
            return None

    @staticmethod
    def _title_looks_legit(meta_title: str, filename_stem: str) -> bool:
        """Le titre des métadonnées PDF est-il fiable (vs titre de template) ?

        Fiable s'il a au moins un mot significatif (>=4 caractères) en commun avec le
        nom de fichier, et s'il n'est pas un titre de gabarit connu.
        """
        low = meta_title.lower().strip()
        if low in _BAD_META_TITLES:
            return False
        if low.startswith(("microsoft word", "untitled", "document")):
            return False

        def words(s: str) -> set[str]:
            return {w.lower() for w in re.findall(r"\w{4,}", s)}

        return bool(words(meta_title) & words(filename_stem))

    def _convert_page(self, page, page_num: int, doc_id: str) -> tuple[bool, str]:
        """Convertit une page en markdown. Retourne (a_une_couche_texte, markdown)."""
        parts: list[str] = []
        try:
            text = (page.get_text("text") or "").strip()
        except Exception:  # noqa: BLE001
            text = ""

        has_text = bool(text)
        if text:
            parts.append(f"## Page {page_num}")
            parts.append("")
            parts.append(text)

        if self.enable_images:
            image_refs = self._extract_page_images(page, page_num, doc_id)
            if image_refs:
                if not has_text:
                    parts.append(f"## Page {page_num}")
                    parts.append("")
                else:
                    parts.append("")
                parts.extend(image_refs)

        return has_text, "\n".join(parts)

    def _extract_page_images(self, page, page_num: int, doc_id: str) -> list[str]:
        """Extrait, stocke et décrit les images d'une page ; retourne les blocs markdown."""
        results: list[str] = []
        try:
            image_list = page.get_images(full=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_images a échoué page %d : %s", page_num, exc)
            return []

        for img_info in image_list:
            xref = img_info[0]
            try:
                base = page.parent.extract_image(xref)
            except Exception as exc:  # noqa: BLE001
                logger.debug("extract_image xref=%s a échoué : %s", xref, exc)
                continue

            img_bytes = base.get("image")
            ext = "." + (base.get("ext") or "png").lower()
            width = base.get("width", 0)
            height = base.get("height", 0)

            if not img_bytes:
                continue
            if len(img_bytes) < self.MIN_IMAGE_BYTES:
                continue
            if width < self.MIN_IMAGE_WIDTH or height < self.MIN_IMAGE_HEIGHT:
                continue

            description = self._describe_with_cache(img_bytes, ext)
            if not description:
                # Sans description, l'image n'a aucune valeur sémantique : on ne
                # l'insère pas dans le markdown (une ré-ingestion la récupérera).
                logger.debug("Image page %d sans description Vision : skip", page_num)
                continue
            if is_decorative_image(description):
                logger.debug("Image page %d décorative : skip", page_num)
                continue

            stored = self.image_store.save(self.collection, doc_id, img_bytes, extension=ext)
            results.append(build_image_block(description, stored.reference))

        return results

    def _describe_with_cache(self, img_bytes: bytes, ext: str) -> str:
        if not self.vision_describer:
            return ""

        img_hash = hashlib.sha256(img_bytes).hexdigest()
        if self.vision_cache_dir:
            cache_file = self.vision_cache_dir / f"{img_hash}.json"
            if cache_file.exists():
                try:
                    return json.loads(cache_file.read_text(encoding="utf-8"))["description"]
                except Exception:  # noqa: BLE001
                    pass

        mime = mime_for_ext(ext)
        try:
            description = self.vision_describer.describe(img_bytes, mime)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Vision a échoué : %s", exc)
            return ""

        description = (description or "").strip()
        if description and self.vision_cache_dir:
            try:
                self.vision_cache_dir.mkdir(parents=True, exist_ok=True)
                (self.vision_cache_dir / f"{img_hash}.json").write_text(
                    json.dumps({"description": description}, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception:  # noqa: BLE001
                pass
        return description

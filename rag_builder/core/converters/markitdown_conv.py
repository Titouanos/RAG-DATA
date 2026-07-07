"""Converter générique via markitdown (fallback pour les formats sans converter dédié).

Gère les formats bureautiques et texte courants (OOXML, HTML, texte, tableaux, e-book).
En v1 il produit le markdown texte de markitdown seul ; l'extraction riche d'images OOXML
reste au backlog.

Note : l'import de ``markitdown`` est paresseux dans ``convert`` (et non dans le
constructeur comme le faisait le POC), pour ne pas imposer la dépendance au simple
instanciation du registre.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rag_builder.core.converters.base import hash_content, make_doc_id
from rag_builder.core.models import ConvertedDoc

logger = logging.getLogger(__name__)


class MarkitdownConverter:
    """Converter markitdown pour les formats OOXML, HTML, texte, tableaux et e-book.

    Extensions gérées : ``.docx``, ``.pptx``, ``.xlsx``, ``.html``, ``.htm``, ``.txt``,
    ``.md``, ``.csv``, ``.json``, ``.xml``, ``.rtf``, ``.epub``.
    """

    SUPPORTED_EXTS = {
        ".docx",
        ".pptx",
        ".xlsx",
        ".html",
        ".htm",
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".xml",
        ".rtf",
        ".epub",
    }

    def can_handle(self, source: Path) -> bool:
        return source.suffix.lower() in self.SUPPORTED_EXTS

    def convert(self, source: Path) -> ConvertedDoc | None:
        try:
            from markitdown import MarkItDown  # import paresseux
        except ImportError:
            logger.error("markitdown n'est pas installé (pip install 'markitdown[all]').")
            return None

        path = Path(source)
        try:
            result = MarkItDown().convert(str(path))
        except Exception as exc:  # noqa: BLE001
            logger.error("Markitdown a échoué sur %s : %s", path.name, exc)
            return None

        markdown = result.text_content or ""
        if not markdown.strip():
            logger.warning("Conversion vide pour %s", path.name)
            return None

        title = result.title or path.stem
        doc_type = path.suffix.lower().lstrip(".")

        return ConvertedDoc(
            doc_id=make_doc_id(path.name),
            source_name=path.name,
            title=title,
            markdown=markdown,
            content_hash=hash_content(markdown),
            doc_type=doc_type,
            metadata={
                "filename": path.name,
                "filepath": str(path),
            },
        )

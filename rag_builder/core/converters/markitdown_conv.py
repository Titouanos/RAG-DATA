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

    Pour ``.html``/``.htm``/``.md``, les images sont post-traitées (`html_images`) :
    data-URI et fichiers relatifs stockés dans l'ImageStore (→ ``rag-image://``,
    description Vision optionnelle), URLs serveur injoignables retirées.
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

    # Formats dont le markdown peut contenir des balises image à réécrire.
    _IMAGE_REWRITE_EXTS = {".html", ".htm", ".md"}

    def __init__(
        self,
        collection: str = "",
        image_store=None,
        vision_describer=None,
        vision_cache_dir: Path | None = None,
        image_roots: list[Path] | None = None,
        remote_fetcher=None,
    ):
        self.collection = collection
        self.image_store = image_store
        self.vision_describer = vision_describer
        self.vision_cache_dir = vision_cache_dir
        self.image_roots = image_roots
        self.remote_fetcher = remote_fetcher

    def can_handle(self, source: Path) -> bool:
        return source.suffix.lower() in self.SUPPORTED_EXTS

    def convert(self, source: Path) -> ConvertedDoc | None:
        try:
            from markitdown import MarkItDown  # import paresseux
        except ImportError:
            logger.error("markitdown n'est pas installé (pip install 'markitdown[all]').")
            return None

        path = Path(source)
        # Pour le HTML/MD, conserver les data-URI (markitdown les tronque par défaut) :
        # le post-traitement les stocke dans l'ImageStore ou les retire — jamais indexés.
        kwargs = (
            {"keep_data_uris": True}
            if path.suffix.lower() in self._IMAGE_REWRITE_EXTS
            else {}
        )
        try:
            try:
                result = MarkItDown().convert(str(path), **kwargs)
            except TypeError:  # markitdown ancien sans keep_data_uris
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
        doc_id = make_doc_id(path.name)

        # Post-traitement des images (HTML/MD) : stockage des data-URI et fichiers
        # relatifs, nettoyage des URLs serveur. Toujours actif pour purger les base64
        # de l'index, même sans ImageStore.
        if path.suffix.lower() in self._IMAGE_REWRITE_EXTS:
            from rag_builder.core.converters.html_images import HtmlImageRewriter

            rewriter = HtmlImageRewriter(
                self.collection,
                self.image_store,
                vision_describer=self.vision_describer,
                vision_cache_dir=self.vision_cache_dir,
                allowed_roots=self.image_roots,
                remote_fetcher=self.remote_fetcher,
            )
            markdown = rewriter.rewrite(markdown, doc_id=doc_id, base_dir=path.parent)

        return ConvertedDoc(
            doc_id=doc_id,
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

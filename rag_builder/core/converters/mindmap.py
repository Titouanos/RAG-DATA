"""Converter MindManager (.mmap) : arbre de topics -> markdown hiérarchique.

Un ``.mmap`` est une archive ZIP contenant un ``Document.xml`` (le nom exact varie selon
la version) et des ressources binaires. On parcourt l'arbre des ``<Topic>`` et on
reconstruit un markdown où la profondeur du topic devient le niveau de titre, en
préservant les liens hypertextes et les notes XHTML.

Les images bitmap (``AlternateImages`` -> ``bin/UUID.bin``) ne sont extraites, décrites et
insérées que si un ``vision_describer`` **et** un ``image_store`` sont fournis ; sinon le
converter produit le texte seul. L'import de ``lxml`` est paresseux dans ``convert``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
import zipfile
from pathlib import Path

from rag_builder.core.converters._image_md import (
    build_image_block,
    is_decorative_image,
    mime_for_ext,
)
from rag_builder.core.converters.base import hash_content, make_doc_id
from rag_builder.core.models import ConvertedDoc

logger = logging.getLogger(__name__)


class MindManagerConverter:
    """Converter des cartes mentales MindManager ``.mmap``."""

    SUPPORTED_EXTS = {".mmap"}

    def __init__(
        self,
        collection: str,
        vision_describer=None,
        image_store=None,
        vision_cache_dir: Path | None = None,
    ):
        self.collection = collection
        self.vision_describer = vision_describer
        self.image_store = image_store
        self.vision_cache_dir = vision_cache_dir
        # Les images n'entrent dans le markdown que si vision + stockage sont fournis.
        self.enable_images = vision_describer is not None and image_store is not None

    def can_handle(self, source: Path) -> bool:
        return source.suffix.lower() in self.SUPPORTED_EXTS

    def convert(self, source: Path) -> ConvertedDoc | None:
        path = Path(source)
        doc_id = make_doc_id(path.name)
        try:
            with tempfile.TemporaryDirectory() as tmp_str:
                tmp = Path(tmp_str)
                with zipfile.ZipFile(path) as z:
                    z.extractall(tmp)

                xml_candidates = list(tmp.glob("*.xml"))
                xml_candidates += list(tmp.glob("Document.xml"))
                xml_candidates += list(tmp.glob("data/*.xml"))
                seen: set[Path] = set()
                xml_candidates = [x for x in xml_candidates if not (x in seen or seen.add(x))]
                if not xml_candidates:
                    logger.warning("Aucun XML trouvé dans %s", path.name)
                    return None

                markdown = self._parse_mindmap(xml_candidates[0], tmp, path.stem, doc_id)
        except zipfile.BadZipFile:
            logger.error("%s n'est pas un ZIP valide (.mmap corrompu ?)", path.name)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("Erreur conversion MindMap %s : %s", path.name, exc, exc_info=True)
            return None

        if not markdown or not markdown.strip():
            logger.warning("MindMap vide : %s", path.name)
            return None

        return ConvertedDoc(
            doc_id=doc_id,
            source_name=path.name,
            title=path.stem,
            markdown=markdown,
            content_hash=hash_content(markdown),
            doc_type="mmap",
            metadata={
                "filename": path.name,
                "filepath": str(path),
            },
        )

    # ------------------------------------------------------------------
    # Parsing XML
    # ------------------------------------------------------------------

    def _parse_mindmap(self, xml_path: Path, base_dir: Path, doc_title: str, doc_id: str) -> str:
        from lxml import etree  # import paresseux

        try:
            tree = etree.parse(str(xml_path))
        except etree.XMLSyntaxError as exc:
            logger.error("XML invalide : %s", exc)
            return ""

        root = tree.getroot()
        lines: list[str] = [f"# {doc_title}", ""]

        central = self._find_root_topic(root)
        if central is None:
            topics = root.xpath("//*[local-name()='Topic']")
            central = topics[0] if topics else None
        if central is None:
            logger.warning("Aucun Topic trouvé dans le XML MindManager")
            return "\n".join(lines)

        self._walk_topic(central, depth=2, lines=lines, base=base_dir, doc_id=doc_id)
        return "\n".join(lines)

    @staticmethod
    def _find_root_topic(root):
        """Cherche le topic central (patterns variables selon la version)."""
        hits = root.xpath("//*[local-name()='OneTopic']/*[local-name()='Topic']")
        if hits:
            return hits[0]
        hits = root.xpath("//*[local-name()='Map']/*[local-name()='Topic']")
        if hits:
            return hits[0]
        return None

    def _walk_topic(self, topic, depth: int, lines: list[str], base: Path, doc_id: str) -> None:
        """Parcours récursif d'un topic et de ses sous-topics."""
        title = self._get_topic_text(topic)
        heading = "#" * min(depth, 6)
        if title:
            lines.append(f"{heading} {title}")
            lines.append("")

        # Liens hypertextes du topic (juste après le titre pour rester dans le chunk).
        topic_links = self._get_topic_links(topic)
        for url in topic_links:
            lines.append(f"_Lien : {url}_")
            lines.append("")

        note = self._get_topic_notes(topic)
        if note:
            lines.append(note)
            lines.append("")

        for url in self._get_note_links(topic):
            if url not in topic_links:
                lines.append(f"_Lien : {url}_")
                lines.append("")

        if self.enable_images:
            for bin_rel_path in self._get_alternate_images(topic):
                self._process_mmap_image(bin_rel_path, base, lines, doc_id)

        sub_topics = topic.xpath("./*[local-name()='SubTopics']/*[local-name()='Topic']")
        if not sub_topics:
            sub_topics = topic.xpath("./*[local-name()='Topic']")
        for sub in sub_topics:
            self._walk_topic(sub, depth + 1, lines, base, doc_id)

    # ------------------------------------------------------------------
    # Liens
    # ------------------------------------------------------------------

    @staticmethod
    def _get_topic_links(topic) -> list[str]:
        """Liens hypertextes http(s) attachés directement au topic."""
        urls: list[str] = []
        for attr in ("Hyperlink", "URL", "Url", "href"):
            val = topic.get(attr)
            if val and val.strip():
                urls.append(val.strip())

        for tag_local in ("Hyperlink", "Link"):
            for el in topic.xpath(f"./*[local-name()='{tag_local}']"):
                for attr in ("Url", "URL", "Address", "href", "Uri"):
                    val = el.get(attr)
                    if val and val.strip():
                        urls.append(val.strip())
                        break
                else:
                    text = (el.text or "").strip()
                    if text and ("://" in text or text.startswith("/")):
                        urls.append(text)

        clean: list[str] = []
        seen: set[str] = set()
        for u in urls:
            if u.startswith(("http://", "https://")) and u not in seen:
                clean.append(u)
                seen.add(u)
        return clean

    @staticmethod
    def _get_note_links(topic) -> list[str]:
        """Liens ``<a href="https://...">`` trouvés dans les notes XHTML du topic."""
        urls: list[str] = []
        seen: set[str] = set()
        a_elements = topic.xpath("./*[local-name()='NotesGroup']//*[local-name()='a']")
        for a in a_elements:
            for attr in ("href", "Href", "URL", "Url"):
                val = a.get(attr)
                if val:
                    val = val.strip()
                    if val.startswith(("http://", "https://")) and val not in seen:
                        urls.append(val)
                        seen.add(val)
                        break
        return urls

    # ------------------------------------------------------------------
    # Notes XHTML
    # ------------------------------------------------------------------

    def _get_topic_notes(self, topic) -> str:
        """Texte plat des notes du topic (préserve les sauts de ligne logiques)."""
        notes_containers = topic.xpath(
            "./*[local-name()='NotesXhtmlData']"
            " | ./*/*[local-name()='NotesXhtmlData']"
            " | ./*/*/*[local-name()='NotesXhtmlData']"
        )
        if not notes_containers:
            notes_containers = topic.xpath("./*[local-name()='NotesGroup']")
        if not notes_containers:
            notes_containers = topic.xpath("./*[local-name()='Notes']")
        if not notes_containers:
            return ""

        parts = [self._extract_html_text(c) for c in notes_containers]
        parts = [p for p in parts if p]
        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _extract_html_text(element) -> str:
        """Extrait le texte d'un fragment XHTML en préservant les blocs logiques."""
        from lxml import etree as _et

        block_tags = {
            "p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6",
            "blockquote", "pre",
        }
        skip_tags = {"uri", "indexviewweight", "indexviewweightcolumn"}
        out_parts: list[str] = []

        def walk(node) -> None:
            tag = _et.QName(node.tag).localname.lower() if isinstance(node.tag, str) else ""
            if tag in skip_tags:
                return
            if node.text:
                out_parts.append(node.text)
            for child in node:
                walk(child)
                if child.tail:
                    out_parts.append(child.tail)
            if tag in block_tags:
                out_parts.append("\n")
            elif tag in ("td", "th"):
                out_parts.append(" | ")

        walk(element)
        text = "".join(out_parts)
        text = re.sub(r"mmarch://bin/[A-Fa-f0-9-]+\.bin", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        return text.strip()

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    @staticmethod
    def _get_alternate_images(topic) -> list[str]:
        """Chemins relatifs (``bin/UUID.bin``) des images bitmap des notes du topic."""
        uris = topic.xpath("./*[local-name()='NotesGroup']//*[local-name()='Uri']")
        results: list[str] = []
        seen: set[str] = set()
        for uri_el in uris:
            uri_text = (uri_el.text or "").strip()
            if uri_text.startswith("mmarch://"):
                rel = uri_text.replace("mmarch://", "")
                if rel not in seen:
                    seen.add(rel)
                    results.append(rel)
        return results

    def _process_mmap_image(
        self, bin_rel_path: str, base: Path, lines: list[str], doc_id: str
    ) -> None:
        """Décrit et insère une image bitmap du ZIP (les vectoriels MJZ sont ignorés)."""
        img_path = base / bin_rel_path
        if not img_path.exists():
            logger.debug("Image bin introuvable : %s", img_path)
            return
        try:
            img_bytes = img_path.read_bytes()
        except OSError as exc:
            logger.debug("Lecture %s échouée : %s", img_path, exc)
            return

        ext = self._detect_image_ext(img_bytes)
        if not ext:
            # Format vectoriel propriétaire (MJZ, zwmf...) : non lisible par Vision.
            return

        description = self._describe_with_cache(img_bytes, ext)
        if not description:
            return
        if is_decorative_image(description):
            logger.debug("Image mmap %s décorative : skip", img_path.name)
            return

        try:
            stored = self.image_store.save(self.collection, doc_id, img_bytes, extension=ext)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Stockage image échoué : %s", exc)
            lines.append(f"**[Capture d'écran]** {description}")
            lines.append("")
            return

        lines.append(build_image_block(description, stored.reference))
        lines.append("")

    @staticmethod
    def _detect_image_ext(img_bytes: bytes) -> str:
        """Détecte le format bitmap via les magic bytes (``''`` si non supporté)."""
        if len(img_bytes) < 8:
            return ""
        if img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png"
        if img_bytes[:3] == b"\xff\xd8\xff":
            return ".jpg"
        if img_bytes[:4] == b"GIF8":
            return ".gif"
        if img_bytes[:2] == b"BM":
            return ".bmp"
        if img_bytes[:4] == b"RIFF" and img_bytes[8:12] == b"WEBP":
            return ".webp"
        return ""

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

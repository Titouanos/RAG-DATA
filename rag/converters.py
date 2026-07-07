"""Converters : transforment chaque type de source en document markdown.

Chaque converter retourne un ``ConvertedDoc`` contenant :
- le markdown produit
- un identifiant stable (basé sur le chemin/url)
- des metadata (source, titre, etc.)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modèle de données
# ---------------------------------------------------------------------------


@dataclass
class ConvertedDoc:
    """Document converti prêt à être chunké."""

    doc_id: str                   # identifiant stable (ex: hash du path ou de l'URL)
    source: str                   # description lisible (chemin du fichier ou URL)
    title: str                    # titre humain
    markdown: str                 # contenu en markdown
    content_hash: str             # hash du contenu markdown (détection de modif)
    metadata: dict = field(default_factory=dict)


class Converter(Protocol):
    """Interface que chaque converter implémente."""

    def can_handle(self, source: Path | str) -> bool: ...
    def convert(self, source: Path | str) -> ConvertedDoc | None: ...


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------


def _hash_path(p: Path) -> str:
    """Hash stable basé sur le chemin absolu (pour servir d'identifiant doc)."""
    return hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:16]


def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _hash_content(text: str) -> str:
    """Hash du contenu pour détecter les modifications."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sanitize_alt_text(text: str) -> str:
    """Nettoie un alt text d'image markdown pour éviter les casses de rendu.

    Problèmes typiques :
    - newlines : cassent la balise markdown (qui doit tenir sur une ligne)
    - crochets [, ] : ferment prématurément le bloc alt
    - apostrophes ' et guillemets " : certains LLM (Gemini) les HTML-escapent
      en &#39; et &quot; quand ils recopient la balise, ce qui rend le alt
      illisible et peut casser le rendu côté OpenWebUI
    - markdown imbriqué (**gras**, *italique*, listes `* item`) : Gemini Vision
      retourne souvent des descriptions très structurées avec du markdown
      dedans. Ça embrouille les LLM qui recopient la balise et casse les rendus.
      On vire tout le markdown du alt text pour le rendre "texte plat".
    - longueur : un alt text trop long (>200 chars) bloque Gemini qui peut
      tronquer la réponse au milieu, ou faire douter de la fin de la balise.
    """
    if not text:
        return ""
    # Aplatir les newlines
    text = text.replace("\n", " ").replace("\r", " ")
    # Virer le markdown bold/italic et les puces de listes
    # **bold** -> bold, *italic* -> italic, les puces "* ... * ..." -> ", ..."
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # **bold**
    text = re.sub(r"\*([^*]+)\*", r"\1", text)      # *italic* / puces de liste
    text = re.sub(r"#+\s*", "", text)               # # titres
    text = re.sub(r"`+", "", text)                  # `code`
    # Remplacer crochets pour ne pas casser la syntaxe markdown
    text = text.replace("[", "(").replace("]", ")")
    # Remplacer apostrophes droites par typographiques (safe partout)
    text = text.replace("\u0027", "\u2019")  # ' -> '
    # Remplacer guillemets droits par typographiques
    text = text.replace("\u0022", "\u00ab")  # " -> «
    # Compacter les espaces multiples
    text = re.sub(r"\s+", " ", text).strip()
    # Cap dur à 150 chars : suffisant pour le retrieval sémantique,
    # assez court pour ne pas perturber la génération du LLM en aval.
    if len(text) > 150:
        # Couper à 147 chars sur un espace si possible (pas en plein mot)
        cut = text[:147]
        last_space = cut.rfind(" ")
        if last_space > 100:  # garde au moins ~100 chars
            cut = cut[:last_space]
        text = cut + "..."
    return text


def _build_image_block(description: str, image_reference: str) -> str:
    """Construit un bloc markdown qui combine description complète + balise image.

    Format :
        *Description complète riche en mots-clés sémantiques...*

        ![alt court](rag-image://...)

    Pourquoi cette structure :
    - La description complète (en italique) est indexée par ChromaDB → quand
      tu poses une question sémantiquement proche, ce chunk remonte.
    - La balise image a un alt court (~80 chars) → le LLM peut la recopier
      sans tronquer la réponse ni se perdre dans un alt trop long.

    Note : la description peut contenir du markdown (Vision retourne souvent
    des **gras**, des * listes). On nettoie tout ça pour avoir du texte plat.
    """
    if not description:
        return ""

    # Nettoyer le markdown imbriqué (Vision retourne souvent du markdown structuré)
    rich_desc = description.replace("\n", " ").replace("\r", " ")
    rich_desc = re.sub(r"\*\*([^*]+)\*\*", r"\1", rich_desc)  # virer le gras
    rich_desc = re.sub(r"\*([^*]+)\*", r"\1", rich_desc)      # virer italique
    rich_desc = re.sub(r"#+\s*", "", rich_desc)               # virer titres
    rich_desc = re.sub(r"`+", "", rich_desc)                  # virer code
    rich_desc = re.sub(r"\s+", " ", rich_desc).strip()

    # Alt court : 80 chars max, pris du début après sanitize
    alt_short = _sanitize_alt_text(description)
    if len(alt_short) > 80:
        cut = alt_short[:77]
        last_space = cut.rfind(" ")
        if last_space > 50:
            cut = cut[:last_space]
        alt_short = cut + "..."

    return f"*{rich_desc}*\n\n![{alt_short}]({image_reference})"


# Patterns qui révèlent qu'une image est décorative (logo, slide de titre,
# bandeau, pictogramme abstrait...). Identifiés à partir des vraies descriptions
# produites par Gemini Vision sur les docs Atlantic.
#
# Les patterns sont cherchés DANS L'ENSEMBLE de la description (pas qu'au début)
# car Gemini commence presque toujours par "La capture d'écran montre..." et
# révèle le caractère décoratif plus loin.
#
# On utilise des patterns précis pour éviter les faux positifs. Par exemple
# "il n'y a pas d'interface logicielle visible" est une formule typique de
# Gemini quand il décrit une image purement décorative (logo, illustration).
_DECORATIVE_PATTERNS = (
    # Phrases finales révélatrices ("aucune action illustrée", "image statique")
    "il n'y a pas d'interface logicielle visible",
    "aucune action spécifique n'est illustrée",
    "aucune action spécifique (clic",
    "il s'agit d'un visuel statique",
    "l'image est purement illustrative",
    "l'image semble être une simple représentation graphique",
    "l'image est une simple représentation graphique",
    "purement illustrative et ne montre aucune action",
    "ne montre aucune action spécifique",
    # Sujet principal qui est un logo ou une marque
    "montre le logo de",
    "présente le logo de",
    "présente le logotype",
    "image de marque",
    # Visuels promotionnels
    "présente un visuel promotionnel",
    "visuel promotionnel ou informatif",
    "visuel statique",
    "page de remerciement",
    "page de conclusion",
    "diapositive de remerciement",
    "diapositive de conclusion",
    # Slides de titre / page de garde
    "diapositive de titre",
    "diapositive d'introduction",
    "slide de titre",
    "page de garde",
    "page de couverture",
    "présente une diapositive de présentation intitulée",
    # Bandeaux et illustrations vectorielles abstraites
    "illustration vectorielle représentant",
    "illustration vectorielle stylisée",
    "scène industrielle stylisée",
    "fond décoratif",
    "présente une vignette graphique",
    "aucun logiciel, interface",
    # Bandeaux décoratifs Atlantic (slash, barre, élément graphique sans contenu)
    "il n'y a aucun texte lisible",
    "aucun texte n'est visible",
    "aucun texte lisible",
    "aucun élément textuel",
    "ne contient aucun texte",
    "il s'agit d'un élément graphique",
    "élément graphique abstrait",
    "élément décoratif",
    "forme géométrique",
    "trait diagonal",
    "barre oblique",
    "ligne diagonale verte",
    "présente un fond blanc",
    "image quasi vide",
    "image principalement blanche",
    "image presque entièrement blanche",
    "espace vide avec",
    "fond blanc avec une simple",
    "fond blanc avec un",
    "fond blanc traversé",
    "un simple trait",
    "une simple forme",
)


def _is_decorative_image(description: str) -> bool:
    """Détecte si la description Vision suggère une image décorative.

    Cherche des patterns révélateurs (phrases que Gemini Vision utilise
    typiquement quand il décrit une image sans valeur informative pour le RAG :
    logos, slides de titre, bandeaux, pictogrammes...).

    Filtrage conservateur : si une image a des patterns mixtes (genre une vraie
    capture qui contient un logo + une interface), Gemini va décrire l'interface
    en priorité et la phrase "il n'y a pas d'interface visible" ne sera PAS
    présente — l'image sera donc gardée.
    """
    if not description:
        return False
    desc_lower = description.lower()
    return any(p in desc_lower for p in _DECORATIVE_PATTERNS)





class LegacyOfficeConverter:
    """Converter pour les anciens formats Office (.doc, .xls, .ppt).

    Ces formats binaires ne sont pas gérés par markitdown. On utilise donc
    Word / Excel / PowerPoint installés sur la machine (Windows uniquement)
    via COM automation pour les convertir en .docx / .xlsx / .pptx, puis on
    passe le fichier converti à markitdown.

    Le fichier converti est mis en cache dans storage/office_cache/ par hash
    du fichier source : si le doc d'origine n'a pas changé, on saute la
    conversion COM (qui est lente : ~2-5s par fichier).

    Si pywin32 n'est pas installé ou si le converter Office ne démarre pas
    (ex. Linux), ce converter retourne None et le fichier est skippé.
    """

    # Format constants Office (cf. doc Microsoft FileFormat enums)
    _WORD_FORMAT_DOCX = 16          # wdFormatDocumentDefault
    _EXCEL_FORMAT_XLSX = 51         # xlOpenXMLWorkbook
    _PPT_FORMAT_PPTX = 24           # ppSaveAsOpenXMLPresentation

    LEGACY_TO_MODERN = {
        ".doc":  (".docx", "word"),
        ".dot":  (".docx", "word"),
        ".xls":  (".xlsx", "excel"),
        ".xlt":  (".xlsx", "excel"),
        ".ppt":  (".pptx", "powerpoint"),
        ".pot":  (".pptx", "powerpoint"),
    }

    def __init__(self, markitdown_converter: "MarkitdownConverter",
                 cache_dir: Path | None = None,
                 office_xml_converter: "OfficeOpenXmlConverter | None" = None):
        """
        :param markitdown_converter: instance de MarkitdownConverter (fallback
                                     si office_xml_converter n'est pas fourni).
        :param cache_dir: dossier où cacher les .docx/.xlsx convertis.
        :param office_xml_converter: si fourni, utilisé à la place de markitdown
                                     pour traiter le fichier .docx/.pptx/.xlsx
                                     issu de la conversion COM. Permet
                                     l'extraction d'images.
        """
        self._markitdown = markitdown_converter
        self._office_xml = office_xml_converter
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        # On vérifie pywin32 paresseusement (pas au boot, pour pas planter sous Linux)
        self._pywin32_checked = False
        self._pywin32_available = False

    def can_handle(self, source: Path | str) -> bool:
        if not isinstance(source, Path):
            return False
        return source.suffix.lower() in self.LEGACY_TO_MODERN

    def convert(self, source: Path | str) -> ConvertedDoc | None:
        path = Path(source)
        ext = path.suffix.lower()
        modern_ext, app = self.LEGACY_TO_MODERN[ext]

        # 1. Cache : si le fichier converti existe déjà, on le réutilise
        cached_modern = self._cached_path(path, modern_ext)
        if cached_modern and cached_modern.exists():
            logger.info("Fichier %s déjà converti dans le cache : %s",
                        path.name, cached_modern.name)
            return self._convert_modern_file(cached_modern, original=path)

        # 2. Vérifier pywin32 disponible
        if not self._ensure_pywin32():
            logger.error(
                "Le fichier %s nécessite une conversion via Word/Excel/PowerPoint, "
                "mais pywin32 n'est pas installé ou Office n'est pas disponible. "
                "Installe pywin32 (pip install pywin32) ou ouvre le fichier dans "
                "Office et enregistre-le manuellement en .docx/.xlsx/.pptx.",
                path.name,
            )
            return None

        # 3. Conversion via COM
        try:
            converted = self._convert_via_office(path, app, modern_ext)
        except Exception as exc:
            logger.error(
                "Conversion Office échouée pour %s : %s. "
                "Astuce : ouvre le fichier dans Office et enregistre-le manuellement "
                "en .docx/.xlsx/.pptx.",
                path.name, exc,
            )
            return None

        if not converted or not converted.exists():
            return None

        # 4. Conversion en ConvertedDoc (avec extraction d'images si possible)
        return self._convert_modern_file(converted, original=path)

    def _convert_modern_file(self, modern_path: Path,
                              original: Path) -> ConvertedDoc | None:
        """Convertit le fichier moderne en ConvertedDoc en utilisant le
        converter approprié (OfficeOpenXmlConverter si dispo, sinon markitdown).

        Les sources/metadata pointent toujours vers le fichier ORIGINAL
        (.doc, .ppt, .xls) pour que l'utilisateur retrouve son fichier d'origine.
        """
        # Préférer OfficeOpenXmlConverter pour avoir l'extraction d'images
        if self._office_xml is not None:
            result = self._office_xml.convert(modern_path)
        else:
            result = self._markitdown.convert(modern_path)

        if not result:
            return None

        # Rebuild avec le path et le doc_id basés sur l'ORIGINAL
        # (sinon ré-ingestion = nouveau doc_id à chaque fois car cached_modern
        #  contient un hash dans le nom)
        return ConvertedDoc(
            doc_id=f"file_{_hash_path(original)}",
            source=str(original),
            title=result.title or original.stem,
            markdown=result.markdown,
            content_hash=_hash_content(result.markdown),
            metadata={
                "type": original.suffix.lower().lstrip("."),
                "filename": original.name,
                "filepath": str(original),
                "converted_via": "legacy_office",
            },
        )

    def _ensure_pywin32(self) -> bool:
        """Check une fois que pywin32 est installé (lazy)."""
        if self._pywin32_checked:
            return self._pywin32_available
        self._pywin32_checked = True
        try:
            import win32com.client  # noqa: F401
            self._pywin32_available = True
        except ImportError:
            self._pywin32_available = False
        return self._pywin32_available

    def _cached_path(self, source: Path, modern_ext: str) -> Path | None:
        """Chemin du fichier converti dans le cache (None si cache désactivé).

        Format : <nom_original>__<hash_court><.docx|.xlsx|.pptx>
        Le hash court (8 chars) est basé sur taille+mtime → si le .doc change,
        le hash change et un nouveau .docx est créé (l'ancien reste dans le
        cache mais devient ignoré).
        """
        if not self.cache_dir:
            return None
        st = source.stat()
        # Empreinte basée sur nom + taille + mtime → si le .doc change, on retransforme
        key = f"{source.name}:{st.st_size}:{int(st.st_mtime)}"
        h = hashlib.sha256(key.encode()).hexdigest()[:8]
        # Garder le nom original lisible, en nettoyant les caractères problématiques
        safe_stem = re.sub(r"[^\w\-. ]", "_", source.stem)
        return self.cache_dir / f"{safe_stem}__{h}{modern_ext}"

    # ------------------------------------------------------------------
    # Conversion COM proprement dite (Windows / Office requis)
    # ------------------------------------------------------------------

    def _convert_via_office(self, source: Path, app: str,
                             modern_ext: str) -> Path | None:
        """Lance Word / Excel / PowerPoint en arrière-plan pour la conversion."""
        import win32com.client
        import pythoncom

        # Le chemin de sortie : cache si dispo, sinon fichier temporaire
        if self.cache_dir:
            output = self._cached_path(source, modern_ext)
        else:
            output = source.with_suffix(modern_ext + ".tmp")

        output_abs = str(output.resolve())
        source_abs = str(source.resolve())

        logger.info("Conversion Office (%s) : %s → %s", app, source.name, output.name)

        pythoncom.CoInitialize()
        try:
            if app == "word":
                self._convert_word(win32com.client, source_abs, output_abs)
            elif app == "excel":
                self._convert_excel(win32com.client, source_abs, output_abs)
            elif app == "powerpoint":
                self._convert_powerpoint(win32com.client, source_abs, output_abs)
            else:
                return None
        finally:
            pythoncom.CoUninitialize()

        return output if output.exists() else None

    @classmethod
    def _convert_word(cls, win32com_client, source: str, output: str):
        word = win32com_client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = False
        try:
            doc = word.Documents.Open(source, ReadOnly=True)
            try:
                doc.SaveAs2(output, FileFormat=cls._WORD_FORMAT_DOCX)
            finally:
                doc.Close(SaveChanges=False)
        finally:
            word.Quit()

    @classmethod
    def _convert_excel(cls, win32com_client, source: str, output: str):
        excel = win32com_client.DispatchEx("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False
        try:
            wb = excel.Workbooks.Open(source, ReadOnly=True)
            try:
                wb.SaveAs(output, FileFormat=cls._EXCEL_FORMAT_XLSX)
            finally:
                wb.Close(SaveChanges=False)
        finally:
            excel.Quit()

    @classmethod
    def _convert_powerpoint(cls, win32com_client, source: str, output: str):
        ppt = win32com_client.DispatchEx("PowerPoint.Application")
        # PowerPoint ne supporte pas Visible=False sur toutes les versions,
        # on le met "minimisé" si possible, sinon on le laisse afficher en arrière-plan
        try:
            ppt.WindowState = 2  # ppWindowMinimized
        except Exception:
            pass
        try:
            pres = ppt.Presentations.Open(source, ReadOnly=True, WithWindow=False)
            try:
                pres.SaveAs(output, FileFormat=cls._PPT_FORMAT_PPTX)
            finally:
                pres.Close()
        finally:
            ppt.Quit()


# ---------------------------------------------------------------------------
# PDF converter (extraction texte + images via PyMuPDF)
# ---------------------------------------------------------------------------


class PdfConverter:
    """Converter dédié aux PDF qui extrait texte ET images.

    - PyMuPDF extrait les images embarquées de chaque page.
    - Chaque image est stockée via ImageStore et décrite par Gemini Vision.
    - Le markdown produit contient des références ![desc](rag-image://...) qui
      survivent au chunking et seront réécrites en URLs HTTP par le MCP server.
    - Cache des descriptions Vision par hash bytes.
    """

    SUPPORTED_EXTS = {".pdf"}

    # Filtres pour ignorer les images décoratives
    MIN_IMAGE_WIDTH = 80
    MIN_IMAGE_HEIGHT = 60
    MIN_IMAGE_BYTES = 2000

    def __init__(self, image_store=None, vision_describer=None,
                 vision_cache_dir: Path | None = None,
                 enable_image_extraction: bool = True):
        self.image_store = image_store
        self.vision_describer = vision_describer
        self.vision_cache_dir = vision_cache_dir
        self.enable_images = (
            enable_image_extraction and image_store is not None
        )

    def can_handle(self, source: Path | str) -> bool:
        if not isinstance(source, Path):
            return False
        return source.suffix.lower() in self.SUPPORTED_EXTS

    def convert(self, source: Path | str) -> ConvertedDoc | None:
        path = Path(source)
        try:
            import pymupdf
        except ImportError:
            try:
                import fitz as pymupdf
            except ImportError:
                logger.error(
                    "PyMuPDF n'est pas installé (pip install pymupdf). "
                    "Désactive l'extraction d'images PDF dans config.yaml ou installe-le."
                )
                return None

        doc_id = f"file_{_hash_path(path)}"

        try:
            pdf = pymupdf.open(str(path))
        except Exception as exc:
            logger.error("Impossible d'ouvrir %s : %s", path.name, exc)
            return None

        markdown_parts: list[str] = []
        # Par défaut on prend le nom du fichier (toujours fiable)
        title = path.stem

        # Le titre des metadata PDF est souvent du bruit (template Word par
        # défaut, nom du gabarit, "Ethan Frome", "Document1", etc.). On ne
        # l'utilise QUE si :
        #   - il existe et n'est pas vide
        #   - ET il contient au moins un mot significatif (>= 4 chars) en
        #     commun avec le nom de fichier (sinon il est probablement bidon).
        meta = pdf.metadata or {}
        meta_title = (meta.get("title") or "").strip()
        if meta_title and self._title_looks_legit(meta_title, path.stem):
            title = meta_title

        markdown_parts.append(f"# {title}")
        markdown_parts.append("")

        page_count = pdf.page_count if hasattr(pdf, "page_count") else None

        try:
            for page_num, page in enumerate(pdf, start=1):
                page_md = self._convert_page(page, page_num, doc_id)
                if page_md.strip():
                    markdown_parts.append(page_md)
                    markdown_parts.append("")
        finally:
            pdf.close()

        markdown = "\n".join(markdown_parts).strip()
        if not markdown:
            logger.warning("PDF vide après conversion : %s", path.name)
            return None

        return ConvertedDoc(
            doc_id=doc_id,
            source=str(path),
            title=title,
            markdown=markdown,
            content_hash=_hash_content(markdown),
            metadata={
                "type": "pdf",
                "filename": path.name,
                "filepath": str(path),
                "pages": page_count,
            },
        )

    @staticmethod
    def _title_looks_legit(meta_title: str, filename_stem: str) -> bool:
        """Heuristique : un titre extrait des metadata PDF est-il fiable ?

        On considère qu'il l'est si il a un mot significatif (>=4 chars) en
        commun avec le nom de fichier — sinon il est probablement le titre
        d'un template Word par défaut ("Ethan Frome" qui est le titre de
        gabarit de Word, "Document1", "Untitled"...).
        """
        # Filtres absolus : titres typiques de template
        bad_titles = {
            "ethan frome", "document", "document1", "untitled",
            "microsoft word", "presentation1", "book1", "classeur1",
        }
        if meta_title.lower().strip() in bad_titles:
            return False
        if meta_title.lower().startswith(("microsoft word", "untitled", "document")):
            return False

        # Sinon : au moins un mot significatif en commun avec le nom de fichier
        def words(s):
            return {w.lower() for w in re.findall(r"\w{4,}", s)}

        meta_words = words(meta_title)
        file_words = words(filename_stem)
        return bool(meta_words & file_words)

    def _convert_page(self, page, page_num: int, doc_id: str) -> str:
        parts: list[str] = []

        try:
            text = page.get_text("text") or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            parts.append(f"## Page {page_num}")
            parts.append("")
            parts.append(text)

        if self.enable_images:
            image_refs = self._extract_page_images(page, page_num, doc_id)
            if image_refs:
                parts.append("")
                parts.extend(image_refs)

        return "\n".join(parts)

    def _extract_page_images(self, page, page_num: int, doc_id: str) -> list[str]:
        """Extrait les images de la page, les stocke et retourne les markdown refs."""
        if self.image_store is None:
            return []

        results: list[str] = []
        try:
            image_list = page.get_images(full=True)
        except Exception as exc:
            logger.debug("get_images a échoué page %d : %s", page_num, exc)
            return []

        for img_info in image_list:
            xref = img_info[0]
            try:
                base = page.parent.extract_image(xref)
            except Exception as exc:
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

            stored = self.image_store.save(doc_id, img_bytes, extension=ext)

            description = self._describe_with_cache(img_bytes, ext)
            if not description:
                # Pas de description (Vision a échoué ou pas dispo).
                # On garde l'image sur disque mais on ne l'insère PAS dans le
                # markdown : sans description elle n'a aucune valeur sémantique
                # et un alt text générique embrouille le LLM. Une ré-ingestion
                # ultérieure (quand le cache Vision sera complet) la récupérera.
                logger.debug(
                    "Image page %d sans description Vision : skip dans le markdown",
                    page_num,
                )
                continue

            if _is_decorative_image(description):
                logger.debug(
                    "Image page %d décorative (description: %s...) : skip",
                    page_num, description[:60],
                )
                continue

            block = _build_image_block(description, stored.reference)
            results.append(block)

        return results

    def _describe_with_cache(self, img_bytes: bytes, ext: str) -> str:
        if not self.vision_describer:
            return ""

        img_hash = hashlib.sha256(img_bytes).hexdigest()
        if self.vision_cache_dir:
            cache_file = self.vision_cache_dir / f"{img_hash}.json"
            if cache_file.exists():
                try:
                    import json as _json
                    return _json.loads(cache_file.read_text(encoding="utf-8"))["description"]
                except Exception:
                    pass

        mime = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
        }.get(ext, "image/png")

        try:
            description = self.vision_describer(img_bytes, mime)
        except Exception as exc:
            logger.debug("Vision a échoué : %s", exc)
            return ""

        description = (description or "").strip()
        if description and self.vision_cache_dir:
            try:
                import json as _json
                self.vision_cache_dir.mkdir(parents=True, exist_ok=True)
                (self.vision_cache_dir / f"{img_hash}.json").write_text(
                    _json.dumps({"description": description}, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception:
                pass
        return description


# ---------------------------------------------------------------------------
# Office Open XML (.docx, .pptx, .xlsx) avec extraction d'images
# ---------------------------------------------------------------------------


class OfficeOpenXmlConverter:
    """Converter pour les formats Office Open XML : .docx, .pptx, .xlsx.

    Stratégie :
    - Markitdown convertit le contenu textuel (texte, tableaux, structure).
    - On parse le ZIP en parallèle pour extraire les images de :
        - .docx : word/media/
        - .pptx : ppt/media/
        - .xlsx : xl/media/
    - Les images sont stockées via ImageStore et décrites par Gemini Vision.
    - Les références markdown ![](rag-image://...) sont **appendées en fin
      de doc** dans une section dédiée.

    Limitation : les images ne sont pas insérées pile à leur position dans
    le texte (ce serait beaucoup plus complexe). Le LLM s'en débrouille via
    leur description Vision : il sait quelle image illustrer dans sa réponse.

    Si markitdown insérait des images en `data:image/png;base64,...` inline,
    on les supprime (pour éviter les balises base64 cassées dans les chunks)
    puisqu'on a maintenant les images sous forme propre via ImageStore.
    """

    SUPPORTED_EXTS = {".docx", ".pptx", ".xlsx"}

    # Dossier interne contenant les images, par extension
    MEDIA_DIRS = {
        ".docx": "word/media/",
        ".pptx": "ppt/media/",
        ".xlsx": "xl/media/",
    }

    # Mêmes filtres que PdfConverter (cohérent)
    MIN_IMAGE_WIDTH = 80
    MIN_IMAGE_HEIGHT = 60
    MIN_IMAGE_BYTES = 2000

    SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

    # Pattern pour repérer (et remplacer) les placeholders d'images insérés
    # par markitdown lors de la conversion. Markitdown utilise plusieurs formats
    # selon le type de doc :
    #
    #   - .docx : ![](data:image/png;base64,iVBORw...)  ou tronqué ![](data:image/png;base64...)
    #   - .pptx : ![alt](Image63.jpg) ou ![](Picture7.png) — réfs vers les
    #             noms internes du ZIP, qui ne pointent vers rien d'utile
    #   - .xlsx : variable selon les versions
    #
    # On match les deux : balises avec `data:image/` ou balises dont l'URL est
    # un nom de fichier image isolé (pas d'URL complète, pas de `/`, juste un
    # nom de fichier avec extension image). Les vraies URLs comme
    # rag-image://... ou http://... sont préservées.
    PLACEHOLDER_RE = re.compile(
        r"!\[[^\]]*\]\("
        r"(?:"
            # Cas 1 : data:image/... (base64 inline)
            r"[^)]*data:image/[^)]*"
            r"|"
            # Cas 2 : nom de fichier image local (pas de scheme, pas de /, pas de :)
            r"[^/:)\s]+\.(?:png|jpg|jpeg|gif|bmp|webp|emf|wmf|tiff?)"
        r")"
        r"\)",
        re.IGNORECASE,
    )

    # Conservé pour compat (alias)
    BASE64_INLINE_RE = PLACEHOLDER_RE

    def __init__(self, markitdown_converter,
                 image_store=None, vision_describer=None,
                 vision_cache_dir: Path | None = None,
                 enable_image_extraction: bool = True):
        """
        :param markitdown_converter: instance de MarkitdownConverter (réutilisée
                                     pour la conversion texte de fichiers
                                     temporaires en .docx/.pptx/.xlsx).
        :param image_store: ImageStore où sauvegarder les images extraites.
        :param vision_describer: callable(img_bytes, mime) -> description.
        :param vision_cache_dir: cache des descriptions par hash bytes.
        :param enable_image_extraction: désactive l'extraction d'images
                                         (fallback pur markitdown).
        """
        self.markitdown_converter = markitdown_converter
        self.image_store = image_store
        self.vision_describer = vision_describer
        self.vision_cache_dir = vision_cache_dir
        self.enable_images = (
            enable_image_extraction and image_store is not None
        )

    def can_handle(self, source: Path | str) -> bool:
        if not isinstance(source, Path):
            return False
        return source.suffix.lower() in self.SUPPORTED_EXTS

    def convert(self, source: Path | str) -> ConvertedDoc | None:
        path = Path(source)
        ext = path.suffix.lower()
        doc_id = self._make_doc_id(path)

        # 1. Pré-extraction : on prépare la liste ordonnée des images du ZIP
        #    pour pouvoir les insérer aux bonnes positions plus tard.
        #    Note : l'ordre dans le ZIP correspond à l'ordre d'apparition des
        #    images dans le doc (vrai pour Word, PowerPoint, Excel).
        ordered_image_refs: list[str] = []
        if self.enable_images:
            ordered_image_refs = self._prepare_ordered_image_refs(path, ext, doc_id)

        # 2. Conversion texte via markitdown
        try:
            from markitdown import MarkItDown
        except ImportError:
            logger.error("markitdown n'est pas installé.")
            return None

        try:
            md = MarkItDown()
            result = md.convert(str(path))
        except Exception as exc:
            logger.error("Markitdown a échoué sur %s : %s", path.name, exc)
            return None

        markdown = (result.text_content or "").strip()
        title = (result.title or path.stem).strip() or path.stem

        # 3. Remplacer les placeholders base64 de markitdown par les vraies
        #    balises markdown ![desc](rag-image://...) dans l'ordre. Si
        #    markitdown a inséré plus de placeholders qu'on a d'images valides
        #    (ex: petites images filtrées), les placeholders restants sont
        #    juste supprimés.
        if ordered_image_refs:
            markdown = self._inject_image_refs(markdown, ordered_image_refs)
        else:
            # Pas d'image valide : on supprime les placeholders cassés
            markdown = self.BASE64_INLINE_RE.sub("", markdown)

        # 4. Nettoyer lignes vides excessives
        markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()

        # 4 bis. Pour les PPTX uniquement : exporter chaque slide en PNG via
        # PowerPoint COM, faire Vision sur chaque PNG, et injecter la description
        # juste après le marqueur "<!-- Slide number: N -->". Cela permet de
        # capturer le contenu visuel des slides denses (grilles, schémas,
        # arborescences) que Vision ne capte pas correctement quand on lui
        # passe juste une image embarquée isolée.
        if ext == ".pptx" and self.enable_images and self.image_store is not None:
            try:
                markdown = self._enrich_pptx_with_slide_screenshots(
                    path, markdown, doc_id,
                )
            except Exception as exc:
                logger.warning(
                    "Export slides entières échoué pour %s : %s",
                    path.name, exc,
                )

        # 5. Ajouter un titre H1 en tête si absent
        if not markdown.startswith("#"):
            markdown = f"# {title}\n\n{markdown}"

        if not markdown:
            logger.warning("Doc Office vide après conversion : %s", path.name)
            return None

        return ConvertedDoc(
            doc_id=doc_id,
            source=str(path),
            title=title,
            markdown=markdown,
            content_hash=_hash_content(markdown),
            metadata={
                "type": ext.lstrip("."),
                "filename": path.name,
                "filepath": str(path),
            },
        )

    def _enrich_pptx_with_slide_screenshots(self, path: Path,
                                              markdown: str,
                                              doc_id: str) -> str:
        """Pour un PPTX : exporte chaque slide en PNG via PowerPoint COM, fait
        passer chaque PNG par Vision, et injecte la description dans le
        markdown juste après le marqueur ``<!-- Slide number: N -->``.

        Cela permet de capturer le contenu visuel des slides denses
        (arborescences, schémas, grilles avec icônes et codes) que ni le texte
        natif du PPTX ni Vision sur les images embarquées isolées ne captent
        correctement.

        Robustesse :
        - Si PowerPoint COM est indisponible (Linux, pas installé), on saute
          silencieusement et on retourne le markdown tel quel.
        - Si l'export d'une slide échoue, on saute juste cette slide.
        - Le cache Vision déduplique automatiquement par hash bytes.

        :param path: chemin vers le .pptx
        :param markdown: markdown généré par markitdown
        :param doc_id: identifiant du doc pour le stockage des images
        :return: markdown enrichi avec les descriptions de slides entières
        """
        # Exporter toutes les slides en PNG dans un dossier temporaire
        slide_pngs = self._export_pptx_slides_to_png(path)
        if not slide_pngs:
            return markdown

        # Pour chaque slide exportée, faire Vision et préparer la description
        # à injecter. On stocke aussi l'image dans ImageStore pour pouvoir la
        # référencer dans le markdown.
        descriptions: dict[int, str] = {}
        try:
            for slide_num, png_path in slide_pngs.items():
                if not png_path.exists():
                    continue
                try:
                    img_bytes = png_path.read_bytes()
                except Exception as exc:
                    logger.debug("Lecture %s échouée : %s", png_path, exc)
                    continue

                # Filtre taille (slides quasi-vides)
                if len(img_bytes) < 4096:
                    continue

                # Vision avec cache hash bytes
                try:
                    description = self._describe_image_bytes_pptx_slide(
                        img_bytes, ext=".png",
                    )
                except Exception as exc:
                    logger.debug("Vision sur slide %d a échoué : %s",
                                 slide_num, exc)
                    continue

                if not description:
                    continue

                # Skip si décorative (slide titre, "Merci", logo...)
                if _is_decorative_image(description):
                    logger.debug(
                        "Slide %d décorative, skip : %s...",
                        slide_num, description[:60],
                    )
                    continue

                # Stocker l'image dans ImageStore et préparer la balise
                try:
                    stored = self.image_store.save(
                        doc_id, img_bytes, extension=".png",
                    )
                    block = _build_image_block(description, stored.reference)
                    descriptions[slide_num] = block
                except Exception as exc:
                    logger.debug("Stockage slide %d échoué : %s",
                                 slide_num, exc)
                    # Fallback : juste la description en texte
                    descriptions[slide_num] = (
                        f"_Vue d'ensemble de la slide :_ {description}"
                    )
        finally:
            # Nettoyer les fichiers temporaires
            self._cleanup_slide_pngs(slide_pngs)

        if not descriptions:
            return markdown

        # Injecter dans le markdown : pour chaque marqueur "<!-- Slide number: N -->",
        # ajouter la description correspondante juste après.
        def replace(match):
            slide_num = int(match.group(1))
            block = descriptions.get(slide_num)
            if block:
                return f"{match.group(0)}\n\n{block}"
            return match.group(0)

        markdown = re.sub(
            r"<!-- Slide number:\s*(\d+)\s*-->",
            replace,
            markdown,
        )
        logger.info(
            "PPTX %s : %d slides enrichies avec Vision (sur %d exportées)",
            path.name, len(descriptions), len(slide_pngs),
        )
        return markdown

    def _export_pptx_slides_to_png(self, path: Path) -> dict[int, Path]:
        """Exporte chaque slide d'un PPTX en PNG via PowerPoint COM (Windows).

        Retourne un dict {slide_number: png_path}. Les PNG sont créés dans
        un dossier temporaire qu'on nettoie ensuite.

        Sur Linux/macOS ou si PowerPoint n'est pas installé, retourne un dict
        vide (le caller fallback gracefully).
        """
        try:
            import win32com.client
            import pythoncom
        except ImportError:
            logger.debug(
                "win32com/pythoncom indisponible, export slides désactivé",
            )
            return {}

        # Init COM pour ce thread (nécessaire si on n'est pas dans le main thread)
        try:
            pythoncom.CoInitialize()
        except Exception:
            pass

        tmp_dir = Path(tempfile.mkdtemp(prefix="pptx_slides_"))
        result: dict[int, Path] = {}
        ppt_app = None
        pres = None

        try:
            ppt_app = win32com.client.Dispatch("PowerPoint.Application")
            # PowerPoint COM nécessite parfois Visible=True (sinon throw).
            # On accepte la fenêtre brièvement, on la fermera après.
            try:
                ppt_app.Visible = True
            except Exception:
                pass

            pres = ppt_app.Presentations.Open(
                str(path.resolve()),
                ReadOnly=True,
                WithWindow=False,
            )

            # Exporter chaque slide en PNG (1280x720 = HD ratio 16:9)
            for slide in pres.Slides:
                slide_num = slide.SlideNumber
                png_path = tmp_dir / f"slide_{slide_num:03d}.png"
                try:
                    slide.Export(str(png_path), "PNG", 1280, 720)
                    if png_path.exists():
                        result[slide_num] = png_path
                except Exception as exc:
                    logger.debug(
                        "Export slide %d échoué : %s", slide_num, exc,
                    )
        except Exception as exc:
            logger.warning("PowerPoint COM a échoué pour %s : %s",
                           path.name, exc)
        finally:
            # Fermer proprement même en cas d'erreur
            try:
                if pres is not None:
                    pres.Close()
            except Exception:
                pass
            try:
                if ppt_app is not None:
                    ppt_app.Quit()
            except Exception:
                pass

        return result

    @staticmethod
    def _cleanup_slide_pngs(slide_pngs: dict[int, Path]):
        """Supprime les fichiers PNG temporaires + leur dossier parent."""
        if not slide_pngs:
            return
        # Tous les PNG sont dans le même dossier tmp
        try:
            parent = next(iter(slide_pngs.values())).parent
            for p in slide_pngs.values():
                try:
                    p.unlink()
                except Exception:
                    pass
            try:
                parent.rmdir()
            except Exception:
                pass
        except Exception:
            pass

    def _describe_image_bytes_pptx_slide(self, img_bytes: bytes,
                                          ext: str = ".png") -> str:
        """Wrapper Vision sur des bytes d'image avec cache par hash.

        Identique à la logique dans MindManagerConverter mais exposé pour
        les slides PPTX exportées en mémoire.
        """
        if not self.enable_images or not self.vision_describer:
            return ""

        img_hash = hashlib.sha256(img_bytes).hexdigest()
        if self.vision_cache_dir:
            cache_file = self.vision_cache_dir / f"{img_hash}.json"
            if cache_file.exists():
                try:
                    return json.loads(cache_file.read_text(encoding="utf-8"))["description"]
                except Exception:
                    pass

        mime_map = {".png": "image/png", ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg", ".gif": "image/gif"}
        mime = mime_map.get(ext, "image/png")

        try:
            description = self.vision_describer(img_bytes, mime) or ""
        except Exception as exc:
            logger.debug("Vision describer error : %s", exc)
            description = ""

        if self.vision_cache_dir and description:
            self.vision_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.vision_cache_dir / f"{img_hash}.json"
            try:
                cache_file.write_text(
                    json.dumps({"description": description}, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception:
                pass

        return description

    def _make_doc_id(self, path: Path) -> str:
        """Identifiant stable basé sur le chemin (et le nom du cache pour les .doc convertis)."""
        return f"file_{_hash_path(path)}"

    def _prepare_ordered_image_refs(self, path: Path, ext: str,
                                     doc_id: str) -> list[str]:
        """Extrait les images du ZIP et retourne la liste ORDONNÉE des balises
        markdown correspondantes : ['![desc1](rag-image://...)', '![desc2](...)', ...].

        L'ordre est celui de leur apparition dans le ZIP, qui correspond à
        l'ordre où markitdown va les rencontrer en convertissant le doc.

        Si une image est filtrée (trop petite) ou n'a pas pu être décrite
        (Vision a échoué), on insère une chaîne vide à sa position dans la
        liste pour préserver l'alignement avec les placeholders de markitdown
        — la chaîne vide sera traduite en "supprimer le placeholder" au moment
        de l'injection.
        """
        media_prefix = self.MEDIA_DIRS.get(ext)
        if not media_prefix:
            return []

        # Collecter les noms triés numériquement (image1.png, image2.png, ...)
        # Word/PowerPoint nomment les images dans l'ordre d'insertion dans le doc.
        try:
            with zipfile.ZipFile(path) as z:
                media_names = [
                    n for n in z.namelist()
                    if n.startswith(media_prefix)
                    and "." + n.rsplit(".", 1)[-1].lower() in self.SUPPORTED_IMAGE_EXTS
                ]
                media_names.sort(key=self._sort_key_image_name)

                refs: list[str] = []
                for name in media_names:
                    img_ext = "." + name.rsplit(".", 1)[-1].lower()
                    try:
                        img_bytes = z.read(name)
                    except Exception as exc:
                        logger.debug("Lecture %s échouée : %s", name, exc)
                        refs.append("")
                        continue

                    # Filtre taille en bytes
                    if len(img_bytes) < self.MIN_IMAGE_BYTES:
                        refs.append("")
                        continue

                    # Filtre dimensions
                    width, height = self._read_image_dimensions(img_bytes, img_ext)
                    if width and height:
                        if width < self.MIN_IMAGE_WIDTH or height < self.MIN_IMAGE_HEIGHT:
                            refs.append("")
                            continue

                    # Stocker l'image (toujours, même si pas de description)
                    stored = self.image_store.save(doc_id, img_bytes, extension=img_ext)

                    # Décrire via Vision (avec cache)
                    description = self._describe_with_cache(img_bytes, img_ext)
                    if not description:
                        logger.debug(
                            "Image %s sans description Vision : skip dans le markdown",
                            name,
                        )
                        refs.append("")
                        continue

                    if _is_decorative_image(description):
                        logger.debug(
                            "Image %s décorative (description: %s...) : skip",
                            name, description[:60],
                        )
                        refs.append("")
                        continue

                    block = _build_image_block(description, stored.reference)
                    refs.append(block)

        except zipfile.BadZipFile:
            logger.warning("%s n'est pas un ZIP valide", path.name)
            return []
        except Exception as exc:
            logger.error("Erreur extraction images de %s : %s", path.name, exc)
            return []

        return refs

    @staticmethod
    def _sort_key_image_name(name: str):
        """Clé de tri pour 'image1.png', 'image10.png', 'image2.png' :
        on trie par numéro extrait (1, 2, 10) plutôt que lexicographiquement."""
        m = re.search(r"(\d+)", name.rsplit("/", 1)[-1])
        return (int(m.group(1)) if m else 0, name)

    def _inject_image_refs(self, markdown: str, ordered_refs: list[str]) -> str:
        """Remplace dans l'ordre chaque placeholder base64 de markitdown par
        la balise markdown correspondante extraite du ZIP.

        Si plus de placeholders que d'images valides : les placeholders en
        trop sont supprimés.
        Si plus d'images que de placeholders (cas atypique) : les images en
        trop sont appendées en fin de doc dans une section dédiée.
        """
        ref_iter = iter(ordered_refs)
        consumed = [False]  # mutable car capturé par closure

        def _replace(match):
            try:
                ref = next(ref_iter)
            except StopIteration:
                # Plus de refs disponibles : on supprime le placeholder
                return ""
            consumed[0] = True
            # ref peut être "" (image filtrée/sans description) → on supprime
            return ref or ""

        new_markdown = self.BASE64_INLINE_RE.sub(_replace, markdown)

        # Vérifier s'il reste des refs non consommées (cas où markitdown a
        # produit moins de placeholders qu'il y a d'images dans le ZIP, ce
        # qui peut arriver pour certaines images en headers/footers).
        # On les met en fin de doc comme avant pour ne rien perdre.
        leftover = [r for r in ref_iter if r]
        if leftover:
            new_markdown = (
                new_markdown.rstrip()
                + "\n\n## Autres images du document\n\n"
                + "\n\n".join(leftover)
                + "\n"
            )

        return new_markdown

    @staticmethod
    def _read_image_dimensions(img_bytes: bytes, ext: str) -> tuple[int, int]:
        """Lit width/height depuis l'en-tête de l'image, sans dépendre de PIL.

        Supporte PNG et JPEG (95% des cas dans les Office docs). Retourne (0, 0)
        sur les autres formats — l'image est alors gardée par défaut.
        """
        try:
            if ext == ".png" and len(img_bytes) >= 24 and img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
                # PNG : width et height sont en big-endian uint32 aux offsets 16 et 20
                width = int.from_bytes(img_bytes[16:20], "big")
                height = int.from_bytes(img_bytes[20:24], "big")
                return width, height
            if ext in (".jpg", ".jpeg") and img_bytes[:2] == b"\xff\xd8":
                # JPEG : parcourir les markers SOF (Start Of Frame)
                i = 2
                while i < len(img_bytes) - 9:
                    if img_bytes[i] != 0xFF:
                        break
                    marker = img_bytes[i + 1]
                    # SOF0..SOF15 sauf DHT (C4), DAC (CC), DRI (DD)
                    if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                        height = int.from_bytes(img_bytes[i + 5:i + 7], "big")
                        width = int.from_bytes(img_bytes[i + 7:i + 9], "big")
                        return width, height
                    # Sinon avancer en lisant la longueur du segment
                    seg_len = int.from_bytes(img_bytes[i + 2:i + 4], "big")
                    i += 2 + seg_len
        except Exception:
            pass
        return 0, 0

    def _describe_with_cache(self, img_bytes: bytes, ext: str) -> str:
        """Décrit l'image via Gemini Vision, avec cache par hash bytes (idem PdfConverter)."""
        if not self.vision_describer:
            return ""

        img_hash = hashlib.sha256(img_bytes).hexdigest()
        if self.vision_cache_dir:
            cache_file = self.vision_cache_dir / f"{img_hash}.json"
            if cache_file.exists():
                try:
                    return json.loads(cache_file.read_text(encoding="utf-8"))["description"]
                except Exception:
                    pass

        mime = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
        }.get(ext, "image/png")

        try:
            description = self.vision_describer(img_bytes, mime)
        except Exception as exc:
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
            except Exception:
                pass
        return description


# ---------------------------------------------------------------------------
# Markitdown generic (formats simples sans images)
# ---------------------------------------------------------------------------


class MarkitdownConverter:
    """Converter générique via markitdown (fallback pour formats sans extraction d'images).

    NOTE : .pdf, .docx, .pptx, .xlsx sont gérés par leurs converters dédiés
    (PdfConverter, OfficeOpenXmlConverter) qui extraient aussi les images.
    """

    SUPPORTED_EXTS = {
        ".html", ".htm", ".txt", ".md", ".csv",
        ".json", ".xml", ".rtf", ".epub",
    }

    def __init__(self):
        from markitdown import MarkItDown
        self._md = MarkItDown()

    def can_handle(self, source: Path | str) -> bool:
        if not isinstance(source, Path):
            return False
        return source.suffix.lower() in self.SUPPORTED_EXTS

    def convert(self, source: Path | str) -> ConvertedDoc | None:
        path = Path(source)
        try:
            result = self._md.convert(str(path))
        except Exception as exc:
            logger.error("Markitdown a échoué sur %s : %s", path.name, exc)
            return None

        markdown = result.text_content or ""
        if not markdown.strip():
            logger.warning("Conversion vide pour %s", path.name)
            return None

        title = result.title or path.stem

        return ConvertedDoc(
            doc_id=f"file_{_hash_path(path)}",
            source=str(path),
            title=title,
            markdown=markdown,
            content_hash=_hash_content(markdown),
            metadata={
                "type": path.suffix.lower().lstrip("."),
                "filename": path.name,
                "filepath": str(path),
            },
        )


# ---------------------------------------------------------------------------
# MindManager (.mmap) converter
# ---------------------------------------------------------------------------


class MindManagerConverter:
    """Converter pour les fichiers .mmap de MindManager.

    Approche :
    - .mmap est un ZIP. On extrait Document.xml et les images jointes.
    - On parcourt l'arbre des <ap:Topic> et reconstruit un markdown hiérarchique.
    - Les images jointes aux topics sont passées à Gemini Vision pour en obtenir
      une description textuelle, insérée dans le markdown.
    - Les descriptions sont cachées par hash d'image pour éviter de rappeler
      la vision à chaque ingestion.
    """

    SUPPORTED_EXTS = {".mmap"}

    def __init__(self, vision_describer=None, cache_dir: Path | None = None,
                 enable_vision: bool = True, image_store=None):
        """
        :param vision_describer: callable(image_bytes: bytes, mime: str) -> str
                                 décrit une image. Peut être None si enable_vision=False.
        :param cache_dir: dossier où cacher les descriptions d'images (par hash).
        :param enable_vision: désactive la vision (les images sont juste mentionnées).
        :param image_store: ImageStore optionnel ; si fourni, les images sont
                            sauvegardées sur disque et leurs références markdown
                            sont insérées dans le markdown.
        """
        self.vision_describer = vision_describer
        self.cache_dir = cache_dir
        self.enable_vision = enable_vision and vision_describer is not None
        self.image_store = image_store

    def can_handle(self, source: Path | str) -> bool:
        if not isinstance(source, Path):
            return False
        return source.suffix.lower() in self.SUPPORTED_EXTS

    def convert(self, source: Path | str) -> ConvertedDoc | None:
        path = Path(source)
        doc_id = f"mmap_{_hash_path(path)}"
        try:
            with tempfile.TemporaryDirectory() as tmp_str:
                tmp = Path(tmp_str)
                with zipfile.ZipFile(path) as z:
                    z.extractall(tmp)

                # Trouver le fichier XML principal (nom varie selon version)
                xml_candidates = list(tmp.glob("*.xml"))
                xml_candidates += list(tmp.glob("Document.xml"))
                xml_candidates += list(tmp.glob("data/*.xml"))
                # déduplication en gardant l'ordre
                seen = set()
                xml_candidates = [x for x in xml_candidates
                                  if not (x in seen or seen.add(x))]
                if not xml_candidates:
                    logger.warning("Aucun XML trouvé dans %s", path.name)
                    return None

                xml_path = xml_candidates[0]
                markdown = self._parse_mindmap(xml_path, tmp, path.stem, doc_id)
        except zipfile.BadZipFile:
            logger.error("%s n'est pas un ZIP valide (.mmap corrompu ?)", path.name)
            return None
        except Exception as exc:
            logger.error("Erreur conversion MindMap %s : %s", path.name, exc, exc_info=True)
            return None

        if not markdown or not markdown.strip():
            logger.warning("MindMap vide : %s", path.name)
            return None

        return ConvertedDoc(
            doc_id=doc_id,
            source=str(path),
            title=path.stem,
            markdown=markdown,
            content_hash=_hash_content(markdown),
            metadata={
                "type": "mindmap",
                "filename": path.name,
                "filepath": str(path),
            },
        )

    def _parse_mindmap(self, xml_path: Path, base_dir: Path, doc_title: str,
                       doc_id: str) -> str:
        """Parse le XML MindManager et produit un markdown hiérarchique."""
        from lxml import etree

        try:
            tree = etree.parse(str(xml_path))
        except etree.XMLSyntaxError as exc:
            logger.error("XML invalide : %s", exc)
            return ""

        root = tree.getroot()
        # Construire le mapping namespace-agnostique
        # On va utiliser des XPath locaux (sans namespace) pour la robustesse
        lines: list[str] = [f"# {doc_title}", ""]

        # Trouver le topic racine (différents schemas possibles selon version)
        central = self._find_root_topic(root)
        if central is None:
            # Fallback : chercher n'importe quel Topic
            topics = root.xpath("//*[local-name()='Topic']")
            central = topics[0] if topics else None

        if central is None:
            logger.warning("Aucun Topic trouvé dans le XML MindManager")
            return "\n".join(lines)

        self._walk_topic(central, depth=2, lines=lines, base=base_dir, doc_id=doc_id)
        return "\n".join(lines)

    def _find_root_topic(self, root):
        """Cherche le topic central. Essaie plusieurs patterns selon les versions."""
        # Pattern 1 : <OneTopic><Topic>
        hits = root.xpath(
            "//*[local-name()='OneTopic']/*[local-name()='Topic']"
        )
        if hits:
            return hits[0]
        # Pattern 2 : <Map><Topic>
        hits = root.xpath("//*[local-name()='Map']/*[local-name()='Topic']")
        if hits:
            return hits[0]
        return None

    def _walk_topic(self, topic, depth: int, lines: list[str], base: Path,
                    doc_id: str):
        """Parcours récursif des topics."""
        title = self._get_topic_text(topic)
        heading = "#" * min(depth, 6)
        if title:
            lines.append(f"{heading} {title}")
            lines.append("")

        # Liens hypertextes du topic — IMMÉDIATEMENT après le titre pour
        # rester dans le même chunk que le titre lors du chunking.
        # Chercher dans plusieurs emplacements possibles selon les versions
        # MindManager.
        topic_links = self._get_topic_links(topic)
        for url in topic_links:
            lines.append(f"_Lien : {url}_")
            lines.append("")

        # Notes attachées
        note = self._get_topic_notes(topic)
        if note:
            lines.append(note)
            lines.append("")

        # Liens trouvés dans les notes XHTML (souvent <a href="https://...">)
        note_links = self._get_note_links(topic)
        for url in note_links:
            if url not in topic_links:  # éviter doublons
                lines.append(f"_Lien : {url}_")
                lines.append("")

        # Images bitmap des notes (AlternateImages -> bin/UUID.bin)
        # Les notes MindManager contiennent des <img src="mmnotes://...zwmf">
        # vectoriels, mais Mindjet stocke aussi une version PNG bitmap dans
        # <AlternateImages><Uri>mmarch://bin/UUID.bin</Uri></AlternateImages>.
        # On récupère ces PNG car Vision peut les lire.
        for bin_rel_path in self._get_alternate_images(topic):
            self._process_mmap_image(bin_rel_path, base, lines, doc_id)

        # Images "natives" attachées au topic (rarement utilisées dans les
        # nouvelles versions, mais on garde pour compatibilité)
        for img_rel in self._get_topic_images(topic):
            img_path = self._resolve_image(base, img_rel)
            if img_path and img_path.exists():
                self._process_resolved_image(img_path, lines, doc_id)

        # Sous-topics
        sub_topics_containers = topic.xpath(
            "./*[local-name()='SubTopics']/*[local-name()='Topic']"
        )
        # Alternative : certains schemas mettent Topic en enfant direct
        if not sub_topics_containers:
            sub_topics_containers = topic.xpath(
                "./*[local-name()='Topic']"
            )
        for sub in sub_topics_containers:
            self._walk_topic(sub, depth + 1, lines, base, doc_id)

    def _get_topic_links(self, topic) -> list[str]:
        """Extrait les liens hypertextes attachés au topic.

        MindManager peut stocker un lien à plusieurs endroits :
        - Attribut Hyperlink/URL/Url sur le Topic (rare, anciennes versions)
        - Élément enfant <Hyperlink Url="..."> ou <Hyperlink Address="...">
        - Élément enfant <Link href="...">

        On cherche dans tous les enfants directs (pas dans SubTopics).
        """
        urls: list[str] = []

        # Attributs directs sur le topic (legacy)
        for attr in ("Hyperlink", "URL", "Url", "href"):
            val = topic.get(attr)
            if val and val.strip():
                urls.append(val.strip())

        # Éléments enfants <Hyperlink> ou <Link> directement sous le Topic
        # (pas profondément, sinon on capture ceux des sous-topics)
        for tag_local in ("Hyperlink", "Link"):
            els = topic.xpath(f"./*[local-name()='{tag_local}']")
            for el in els:
                # L'URL peut être dans un attribut (Url, URL, Address, href, Uri)
                # ou dans le contenu texte
                for attr in ("Url", "URL", "Address", "href", "Uri"):
                    val = el.get(attr)
                    if val and val.strip():
                        urls.append(val.strip())
                        break
                else:
                    text = (el.text or "").strip()
                    if text and ("://" in text or text.startswith("/")):
                        urls.append(text)

        # Filtrage : garder uniquement les vraies URLs http(s) (pas mmarch://, pas chemins locaux)
        clean = []
        seen = set()
        for u in urls:
            if u.startswith(("http://", "https://")) and u not in seen:
                clean.append(u)
                seen.add(u)
        return clean

    def _get_note_links(self, topic) -> list[str]:
        """Extrait les liens <a href="https://..."> trouvés dans les notes XHTML
        du topic (pas des sous-topics)."""
        urls: list[str] = []
        seen = set()
        # Chercher tous les <a> dans NotesGroup/NotesData/NotesXhtmlData direct
        # (limité au topic courant, pas sous-topics)
        a_elements = topic.xpath(
            "./*[local-name()='NotesGroup']"
            "//*[local-name()='a']"
        )
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

    def _get_alternate_images(self, topic) -> list[str]:
        """Récupère les chemins relatifs (bin/UUID.bin) des images bitmap
        attachées aux notes du topic.

        Format dans MindManager moderne :
            <Topic>
              <NotesGroup>
                <NotesData>
                  <NotesXhtmlData>...HTML avec <img src="mmarch://..."> ...</NotesXhtmlData>
                  <AlternateImages ImageType="urn:mindjet:PngImage">
                    <Uri>mmarch://bin/UUID.bin</Uri>
                  </AlternateImages>
                </NotesData>
              </NotesGroup>
            </Topic>

        On capture TOUTES les références mmarch:// dans NotesGroup/NotesData
        du topic (pas des sous-topics, qui seront traités à part par récursion).
        Le filtrage final (PNG vs MJZ vectoriel) se fait au moment du traitement
        en regardant les magic bytes du fichier — c'est plus fiable que de se
        fier aux attributs ImageType de Mindjet.

        On déduplique et on préserve l'ordre d'apparition.
        """
        # Capture brute : tout Uri mmarch:// dans le NotesGroup du topic.
        # Le `./*[local-name()='NotesGroup']//*` descend dans tout le contenu
        # de NotesGroup mais ne traverse pas la frontière SubTopics car les
        # SubTopics ne sont pas dans NotesGroup.
        uris = topic.xpath(
            "./*[local-name()='NotesGroup']"
            "//*[local-name()='Uri']"
        )

        results = []
        seen = set()
        for uri_el in uris:
            uri_text = (uri_el.text or "").strip()
            if uri_text.startswith("mmarch://"):
                rel = uri_text.replace("mmarch://", "")
                if rel not in seen:
                    seen.add(rel)
                    results.append(rel)
        return results

    def _process_mmap_image(self, bin_rel_path: str, base: Path,
                            lines: list[str], doc_id: str):
        """Traite une image extraite du ZIP MindManager.

        Le ZIP contient un mélange :
        - PNG (vraies captures bitmap, magic = '\\x89PNG')
        - MJZ (format vectoriel Mindjet propriétaire, magic = 'MJZ\\x00' ou 'MJZ\\x07')
        - autres formats vectoriels (zwmf, emf...)

        On filtre via les magic bytes : seuls les formats lisibles par Vision
        passent. Les vectoriels propriétaires sont silencieusement skippés.

        Note : pas de filtre par taille minimale, car certaines captures
        d'icônes utiles peuvent faire moins de 2KB (cf. petites barres d'outils).
        Le filtre _is_decorative_image (basé sur la description Vision) fait
        le vrai tri pertinence.
        """
        img_path = base / bin_rel_path
        if not img_path.exists():
            logger.debug("Image bin introuvable : %s", img_path)
            return
        try:
            img_bytes = img_path.read_bytes()
        except Exception as exc:
            logger.debug("Lecture %s échouée : %s", img_path, exc)
            return

        # Détection format via magic bytes : seul moyen fiable de distinguer
        # les vraies images bitmap des formats vectoriels Mindjet (MJZ).
        ext = self._detect_image_ext(img_bytes)
        if not ext:
            # MJZ ou autre format vectoriel : skip silencieusement
            return

        # Vision (avec cache)
        description = ""
        if self.enable_vision:
            try:
                description = self._describe_image_bytes(img_bytes, ext)
            except Exception as exc:
                logger.debug("Vision a échoué sur %s : %s", img_path.name, exc)

        if not description:
            return

        # Skip si décorative
        if _is_decorative_image(description):
            logger.debug(
                "Image mmap %s décorative (description: %s...) : skip",
                img_path.name, description[:60],
            )
            return

        # Stocker et insérer
        if self.image_store is not None:
            try:
                stored = self.image_store.save(doc_id, img_bytes, extension=ext)
                block = _build_image_block(description, stored.reference)
                lines.append(block)
                lines.append("")
                return
            except Exception as exc:
                logger.debug("Stockage image échoué : %s", exc)

        # Fallback texte
        lines.append(f"**[Capture d'écran]** {description}")
        lines.append("")

    def _process_resolved_image(self, img_path: Path, lines: list[str],
                                 doc_id: str):
        """Traite une image native déjà résolue (depuis _get_topic_images)."""
        description = self._describe_image(img_path)
        if description and _is_decorative_image(description):
            logger.debug(
                "Image mindmap %s décorative (description: %s...) : skip",
                img_path.name, description[:60],
            )
            return
        if self.image_store is not None and description:
            try:
                img_bytes = img_path.read_bytes()
                ext = img_path.suffix or ".png"
                stored = self.image_store.save(doc_id, img_bytes, extension=ext)
                block = _build_image_block(description, stored.reference)
                lines.append(block)
                lines.append("")
                return
            except Exception as exc:
                logger.debug("Stockage image échoué : %s", exc)

        if description:
            lines.append(f"**[Capture d'écran]** {description}")
            lines.append("")
        else:
            lines.append(f"**[Capture d'écran : {img_path.name}]**")
            lines.append("")

    @staticmethod
    def _detect_image_ext(img_bytes: bytes) -> str:
        """Détecte le format d'une image via ses magic bytes.

        Retourne '.png', '.jpg', '.gif', etc. ou '' si format inconnu/non supporté
        par Gemini Vision.
        """
        if len(img_bytes) < 8:
            return ""
        # PNG : 89 50 4E 47 0D 0A 1A 0A
        if img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png"
        # JPEG : FF D8 FF
        if img_bytes[:3] == b"\xff\xd8\xff":
            return ".jpg"
        # GIF : GIF8
        if img_bytes[:4] in (b"GIF8",):
            return ".gif"
        # BMP : BM
        if img_bytes[:2] == b"BM":
            return ".bmp"
        # WEBP : RIFF....WEBP
        if img_bytes[:4] == b"RIFF" and img_bytes[8:12] == b"WEBP":
            return ".webp"
        return ""

    def _describe_image_bytes(self, img_bytes: bytes, ext: str) -> str:
        """Comme _describe_image mais à partir de bytes en mémoire (pas un fichier).

        Réutilise le cache par hash de bytes.
        """
        if not self.enable_vision or not self.vision_describer:
            return ""

        # Hash pour le cache
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        if self.cache_dir:
            cache_file = self.cache_dir / f"{img_hash}.json"
            if cache_file.exists():
                try:
                    return json.loads(cache_file.read_text(encoding="utf-8"))["description"]
                except Exception:
                    pass

        # Mime type
        mime_map = {".png": "image/png", ".jpg": "image/jpeg",
                    ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp"}
        mime = mime_map.get(ext, "image/png")

        try:
            description = self.vision_describer(img_bytes, mime) or ""
        except Exception as exc:
            logger.debug("Vision describer error : %s", exc)
            description = ""

        # Sauver le cache
        if self.cache_dir and description:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{img_hash}.json"
            try:
                cache_file.write_text(
                    json.dumps({"description": description}, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception:
                pass

        return description


    def _get_topic_text(self, topic) -> str:
        """Extrait le texte d'un topic, en testant plusieurs patterns."""
        # Pattern 1 : <Text PlainText="..."/>
        texts = topic.xpath("./*[local-name()='Text']")
        for t in texts:
            plain = t.get("PlainText")
            if plain:
                return plain.strip()
            # Texte dans les enfants <Font>, <Span>, etc.
            inner = "".join(t.itertext()).strip()
            if inner:
                return inner
        # Pattern 2 : attribut direct sur le topic
        t = topic.get("Text") or topic.get("PlainText")
        if t:
            return t.strip()
        return ""

    def _get_topic_notes(self, topic) -> str:
        """Extrait les notes (texte riche) d'un topic.

        Dans MindManager, les notes sont stockées dans un conteneur
        <NotesGroup> ou <NotesXhtmlData> avec du HTML embarqué (paragraphs,
        listes, tableaux, liens, parfois images). On extrait le texte plat
        en préservant les structures sémantiques utiles (listes, paragraphes).

        IMPORTANT : NotesGroup contient souvent NotesXhtmlData en enfant.
        On cherche donc le conteneur le plus profond (NotesXhtmlData) en priorité,
        et on ne fallback sur NotesGroup que s'il n'y en a pas. Sinon on
        dupliquerait le contenu (le NotesGroup retournerait tout son sous-arbre,
        y compris le NotesXhtmlData déjà parsé).
        """
        # Priorité 1 : NotesXhtmlData (le plus précis, dans les versions récentes)
        notes_containers = topic.xpath(
            "./*[local-name()='NotesXhtmlData']"
            " | ./*/*[local-name()='NotesXhtmlData']"
            " | ./*/*/*[local-name()='NotesXhtmlData']"
        )

        # Si pas de NotesXhtmlData, fallback sur NotesGroup direct
        if not notes_containers:
            notes_containers = topic.xpath("./*[local-name()='NotesGroup']")

        # Legacy : Notes simple
        if not notes_containers:
            notes_containers = topic.xpath("./*[local-name()='Notes']")

        if not notes_containers:
            return ""

        # Extraire le texte de chaque conteneur en préservant la structure
        parts = []
        for container in notes_containers:
            text = self._extract_html_text(container)
            if text:
                parts.append(text)

        if not parts:
            return ""

        return "\n\n".join(parts)

    @staticmethod
    def _extract_html_text(element) -> str:
        """Extrait le texte d'un élément contenant du HTML embarqué (XHTML).

        Préserve les sauts de ligne logiques :
        - <p>, <div>, <li>, <tr>, <br> -> nouvelle ligne
        - <ul>, <ol> -> rien (les <li> dedans s'en chargent)
        - liens : on garde l'ancre (texte) mais on supprime les URI mmarch://
          qui sont des références internes binaires du ZIP MindManager.

        Compacte ensuite les espaces multiples.
        """
        from lxml import etree as _et

        # Tags qui imposent un saut de ligne après leur contenu
        block_tags = {
            "p", "div", "li", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6",
            "blockquote", "pre",
        }
        # Tags à ignorer complètement (références binaires, métadonnées)
        skip_tags = {"uri", "indexviewweight", "indexviewweightcolumn"}

        out_parts = []

        def walk(node):
            tag = _et.QName(node.tag).localname.lower() if isinstance(node.tag, str) else ""
            if tag in skip_tags:
                return
            # Texte avant les enfants
            if node.text:
                out_parts.append(node.text)
            for child in node:
                walk(child)
                if child.tail:
                    out_parts.append(child.tail)
            # Saut de ligne après les blocs
            if tag in block_tags:
                out_parts.append("\n")
            elif tag in ("td", "th"):
                # Cellules : séparateur léger
                out_parts.append(" | ")

        walk(element)
        text = "".join(out_parts)
        # Virer les liens binaires mmarch:// (références internes au ZIP)
        text = re.sub(r"mmarch://bin/[A-Fa-f0-9-]+\.bin", "", text)
        # Virer les URLs nues isolées si elles sont du bruit
        # Compacter les espaces et sauts de ligne excessifs
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        return text.strip()

    def _get_topic_images(self, topic) -> list[str]:
        """Liste des chemins relatifs d'images attachées au topic."""
        images = []
        # Images via <Image Path="..."/> ou <Attachment Path="..."/>
        for img in topic.xpath(".//*[local-name()='Image']"):
            src = img.get("Path") or img.get("URL") or img.get("Source")
            if src and self._is_image_ext(src):
                images.append(src)
        for att in topic.xpath(".//*[local-name()='Attachment']"):
            src = att.get("Path") or att.get("URL")
            if src and self._is_image_ext(src):
                images.append(src)
        return images

    @staticmethod
    def _is_image_ext(path: str) -> bool:
        return path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"))

    def _resolve_image(self, base: Path, rel: str) -> Path | None:
        """Résout le chemin d'une image relative dans le ZIP extrait."""
        rel = rel.lstrip("/\\")
        candidates = [
            base / rel,
            base / "attachments" / rel,
            base / "images" / rel,
            base / "Data" / rel,
        ]
        # Recherche par nom de fichier si les chemins ne matchent pas
        for c in candidates:
            if c.exists():
                return c
        # Fallback : scan complet par nom
        name = Path(rel).name
        for found in base.rglob(name):
            return found
        return None

    def _describe_image(self, img_path: Path) -> str:
        """Appelle Gemini Vision (avec cache par hash)."""
        if not self.enable_vision:
            return ""

        img_bytes = img_path.read_bytes()
        img_hash = hashlib.sha256(img_bytes).hexdigest()

        # Cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{img_hash}.json"
            if cache_file.exists():
                try:
                    return json.loads(cache_file.read_text(encoding="utf-8"))["description"]
                except Exception:
                    pass  # cache corrompu, on régénère

        # Détection mime basique
        ext = img_path.suffix.lower()
        mime = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp",
        }.get(ext, "image/png")

        try:
            description = self.vision_describer(img_bytes, mime)
        except Exception as exc:
            logger.warning("Vision a échoué sur %s : %s", img_path.name, exc)
            return ""

        description = (description or "").strip()
        if description and self.cache_dir:
            cache_file = self.cache_dir / f"{img_hash}.json"
            cache_file.write_text(
                json.dumps({"description": description}, ensure_ascii=False),
                encoding="utf-8",
            )
        return description


# ---------------------------------------------------------------------------
# YouTube converter
# ---------------------------------------------------------------------------


YOUTUBE_REGEX = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/|youtube\.com/embed/)"
    r"([A-Za-z0-9_-]{11})",
    re.IGNORECASE,
)


def extract_youtube_id(url: str) -> str | None:
    """Extrait l'ID d'une URL YouTube. None si URL invalide."""
    m = YOUTUBE_REGEX.search(url.strip())
    return m.group(1) if m else None


class YouTubeConverter:
    """Converter pour les URLs YouTube via youtube-transcript-api.

    On récupère la transcription (auto-générée ou manuelle) et on la regroupe
    par tranches de temps pour former des sections markdown avec timestamps.
    """

    def __init__(self, preferred_langs: list[str], group_seconds: int = 60):
        self.preferred_langs = preferred_langs
        self.group_seconds = group_seconds

    def can_handle(self, source: Path | str) -> bool:
        return isinstance(source, str) and extract_youtube_id(source) is not None

    def convert(self, source: Path | str) -> ConvertedDoc | None:
        url = str(source)
        video_id = extract_youtube_id(url)
        if not video_id:
            return None

        try:
            transcript_data, used_lang = self._fetch_transcript(video_id)
        except Exception as exc:
            logger.error("Impossible de récupérer le transcript de %s : %s", url, exc)
            return None

        if not transcript_data:
            return None

        title = f"YouTube — {video_id}"
        markdown = self._format_markdown(transcript_data, video_id, title, used_lang)

        return ConvertedDoc(
            doc_id=f"yt_{video_id}",
            source=url,
            title=title,
            markdown=markdown,
            content_hash=_hash_content(markdown),
            metadata={
                "type": "youtube",
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "lang": used_lang,
            },
        )

    def _fetch_transcript(self, video_id: str):
        """Récupère la meilleure transcription disponible.

        Retourne (segments, lang_used) où segments = [{'text', 'start', 'duration'}, ...].
        Compatible avec différentes versions de youtube-transcript-api.
        """
        from youtube_transcript_api import YouTubeTranscriptApi

        # API récente (>=1.0) : instance + fetch
        if hasattr(YouTubeTranscriptApi, "list") and not hasattr(YouTubeTranscriptApi, "list_transcripts"):
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            transcript = None
            for lang in self.preferred_langs:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    break
                except Exception:
                    continue
            if transcript is None:
                # Prendre n'importe lequel et traduire si possible
                available = list(transcript_list)
                if not available:
                    return None, None
                transcript = available[0]
            fetched = transcript.fetch()
            # fetched peut être un FetchedTranscript (objet) ou liste de dicts
            if hasattr(fetched, "to_raw_data"):
                segments = fetched.to_raw_data()
            elif hasattr(fetched, "__iter__"):
                segments = [
                    {"text": s.text if hasattr(s, "text") else s["text"],
                     "start": s.start if hasattr(s, "start") else s["start"],
                     "duration": s.duration if hasattr(s, "duration") else s.get("duration", 0)}
                    for s in fetched
                ]
            else:
                segments = fetched
            return segments, transcript.language_code

        # API legacy (<1.0) : méthodes statiques
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = None
            for lang in self.preferred_langs:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    break
                except Exception:
                    continue
            if transcript is None:
                available = list(transcript_list)
                if not available:
                    return None, None
                transcript = available[0]
            return transcript.fetch(), transcript.language_code
        except AttributeError:
            # Très ancienne API
            segments = YouTubeTranscriptApi.get_transcript(
                video_id, languages=self.preferred_langs,
            )
            return segments, self.preferred_langs[0]

    def _format_markdown(self, segments, video_id: str, title: str, lang: str) -> str:
        """Regroupe les segments par tranches de temps et formate en markdown."""
        lines = [f"# {title}", "", f"Source : https://www.youtube.com/watch?v={video_id}  ",
                 f"Langue : {lang}", ""]

        # Regrouper par tranches
        groups: list[tuple[float, list[str]]] = []
        current_start = 0.0
        current_texts: list[str] = []

        for seg in segments:
            start = float(seg.get("start", 0))
            text = (seg.get("text") or "").strip().replace("\n", " ")
            if not text:
                continue
            if not current_texts:
                current_start = start
                current_texts.append(text)
                continue
            if start - current_start >= self.group_seconds:
                groups.append((current_start, current_texts))
                current_start = start
                current_texts = [text]
            else:
                current_texts.append(text)
        if current_texts:
            groups.append((current_start, current_texts))

        for start, texts in groups:
            ts = self._format_timestamp(start)
            lines.append(f"## [{ts}] Segment à {ts}")
            lines.append("")
            lines.append(" ".join(texts))
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total = int(seconds)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Video converter (.mp4, .mov, .webm, etc.)
# ---------------------------------------------------------------------------


class VideoConverter:
    """Converter pour les fichiers vidéo locaux.

    Utilise Gemini en multimodal (File API) pour obtenir en un appel :
    - la transcription des paroles avec timestamps
    - la description des éléments visuels (interfaces, commandes, etc.)

    Le résultat est caché par empreinte du fichier (taille + mtime + nom) pour
    éviter de retranscrire inutilement, car c'est coûteux. Si le fichier est
    modifié (mtime change), la transcription est refaite.
    """

    SUPPORTED_EXTS = {
        ".mp4", ".mov", ".webm", ".avi", ".mpeg", ".mpg",
        ".wmv", ".flv", ".3gp", ".3gpp", ".m4v", ".mkv",
    }

    def __init__(self, transcriber=None, cache_dir: Path | None = None,
                 enable: bool = True):
        """
        :param transcriber: callable(video_path) -> str markdown, ex.
                            ``make_video_transcriber(...)`` depuis rag.llm.
        :param cache_dir: dossier de cache des transcriptions (par empreinte fichier).
        :param enable: désactive la transcription (le fichier est ignoré).
        """
        self.transcriber = transcriber
        self.cache_dir = cache_dir
        self.enable = enable and transcriber is not None

    def can_handle(self, source: Path | str) -> bool:
        if not isinstance(source, Path):
            return False
        return source.suffix.lower() in self.SUPPORTED_EXTS

    def convert(self, source: Path | str) -> ConvertedDoc | None:
        path = Path(source)
        if not self.enable:
            logger.warning(
                "Transcription vidéo désactivée (ou pas de transcriber) : skip %s",
                path.name,
            )
            return None

        # 1. Cache par empreinte fichier
        file_fp = self._file_fingerprint(path)
        cached_md = self._read_cache(file_fp)
        if cached_md:
            logger.info("Transcription vidéo depuis cache : %s", path.name)
            markdown = cached_md
        else:
            logger.info(
                "Transcription de la vidéo %s (peut prendre plusieurs minutes)...",
                path.name,
            )
            try:
                markdown = self.transcriber(path)
            except Exception as exc:
                logger.error("Transcription vidéo échouée (%s) : %s",
                             path.name, exc, exc_info=True)
                return None
            if markdown and markdown.strip():
                self._write_cache(file_fp, markdown)

        if not markdown or not markdown.strip():
            logger.warning("Transcription vide pour %s", path.name)
            return None

        # 2. Extraire le titre depuis le premier H1 s'il existe
        title = path.stem
        for line in markdown.splitlines():
            m = re.match(r"^#\s+(.+)$", line.strip())
            if m:
                title = m.group(1).strip()
                break

        # 3. Préfixer avec les métadonnées source
        header = (
            f"Source vidéo : {path.name}  \n"
            f"Chemin : {path}\n\n"
        )
        markdown_full = header + markdown

        return ConvertedDoc(
            doc_id=f"video_{_hash_path(path)}",
            source=str(path),
            title=title,
            markdown=markdown_full,
            content_hash=_hash_content(markdown_full),
            metadata={
                "type": "video",
                "filename": path.name,
                "filepath": str(path),
                "extension": path.suffix.lower().lstrip("."),
            },
        )

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _file_fingerprint(self, path: Path) -> str:
        """Empreinte rapide basée sur nom + taille + mtime.

        Change dès qu'on remplace le fichier → la transcription est refaite.
        """
        st = path.stat()
        key = f"{path.name}:{st.st_size}:{int(st.st_mtime)}"
        return hashlib.sha256(key.encode()).hexdigest()[:20]

    def _read_cache(self, fp: str) -> str | None:
        if not self.cache_dir:
            return None
        cache_file = self.cache_dir / f"{fp}.md"
        if cache_file.exists():
            try:
                return cache_file.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    def _write_cache(self, fp: str, markdown: str):
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{fp}.md").write_text(markdown, encoding="utf-8")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ConverterRegistry:
    """Orchestre les converters et expose une méthode de conversion unique."""

    def __init__(self, converters: list[Converter]):
        self.converters = converters

    def convert(self, source: Path | str) -> ConvertedDoc | None:
        for conv in self.converters:
            if conv.can_handle(source):
                return conv.convert(source)
        logger.debug("Aucun converter pour %s", source)
        return None


# ---------------------------------------------------------------------------
# Scan du dossier data/
# ---------------------------------------------------------------------------


def iter_sources(data_dir: Path) -> Iterator[Path | str]:
    """Itère sur toutes les sources à convertir depuis le dossier data/.

    - Chaque fichier du dossier = une source (récursif)
    - Les fichiers .txt sont inspectés : si une ligne contient une URL YouTube,
      elle devient une source à part entière (au lieu du fichier .txt lui-même).
    """
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        # Si c'est un .txt, check si ça contient des URLs YouTube
        if path.suffix.lower() == ".txt":
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                yield path
                continue
            urls_found = False
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if extract_youtube_id(line):
                    urls_found = True
                    yield line
            if not urls_found:
                # .txt normal, on le traite comme doc
                yield path
        else:
            yield path
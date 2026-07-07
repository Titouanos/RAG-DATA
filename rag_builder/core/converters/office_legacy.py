"""Converter des formats Office legacy (.doc/.dot/.xls/.xlt/.ppt/.pot) via LibreOffice.

Les formats binaires legacy ne sont pas gérés par markitdown. On les convertit en OOXML
(``docx``/``xlsx``/``pptx``) avec LibreOffice en mode headless
(``soffice --headless --convert-to``), puis on délègue le résultat à
:class:`MarkitdownConverter`.

Le fichier OOXML est mis en cache par empreinte (nom + taille + mtime) pour éviter de
relancer la conversion — lente — quand la source n'a pas changé. Le ``doc_id``, le
``source_name`` et le titre sont rebasés sur le fichier **original** afin que l'ingestion
incrémentale reste stable et que l'utilisateur retrouve son fichier d'origine.

Dégradation propre : si ``soffice`` est absent du PATH, le converter logge un warning et
retourne ``None`` (le fichier est simplement ignoré).
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
from pathlib import Path

from rag_builder.core.converters.base import hash_content, make_doc_id
from rag_builder.core.converters.markitdown_conv import MarkitdownConverter
from rag_builder.core.models import ConvertedDoc

logger = logging.getLogger(__name__)

# Timeout (secondes) de la conversion LibreOffice headless.
_SOFFICE_TIMEOUT = 120


class LibreOfficeConverter:
    """Convertit les formats Office legacy en OOXML via LibreOffice puis markitdown."""

    # Extension legacy -> (extension OOXML, filtre --convert-to).
    LEGACY_TO_MODERN = {
        ".doc": (".docx", "docx"),
        ".dot": (".docx", "docx"),
        ".xls": (".xlsx", "xlsx"),
        ".xlt": (".xlsx", "xlsx"),
        ".ppt": (".pptx", "pptx"),
        ".pot": (".pptx", "pptx"),
    }

    def __init__(self, cache_dir: Path, markitdown_converter: MarkitdownConverter):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._markitdown = markitdown_converter

    def can_handle(self, source: Path) -> bool:
        return source.suffix.lower() in self.LEGACY_TO_MODERN

    def convert(self, source: Path) -> ConvertedDoc | None:
        path = Path(source)
        ext = path.suffix.lower()
        modern_ext, target_filter = self.LEGACY_TO_MODERN[ext]

        # 1. Cache : réutiliser l'OOXML déjà converti si présent.
        cached = self._cached_path(path, modern_ext)
        if cached.exists():
            logger.info("Réutilisation du cache LibreOffice : %s", cached.name)
            return self._delegate(cached, original=path)

        # 2. LibreOffice disponible ?
        if shutil.which("soffice") is None:
            logger.warning(
                "%s nécessite LibreOffice (soffice) pour la conversion en OOXML, "
                "mais 'soffice' est introuvable dans le PATH : fichier ignoré.",
                path.name,
            )
            return None

        # 3. Conversion headless dans un dossier temporaire, puis copie vers le cache.
        converted = self._convert_via_soffice(path, target_filter, modern_ext)
        if converted is None:
            return None

        return self._delegate(converted, original=path)

    def _convert_via_soffice(
        self, source: Path, target_filter: str, modern_ext: str
    ) -> Path | None:
        """Lance ``soffice --headless --convert-to`` ; renvoie le chemin caché ou None."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            cmd = [
                "soffice",
                "--headless",
                "--convert-to",
                target_filter,
                "--outdir",
                str(tmp),
                str(source),
            ]
            logger.info("Conversion LibreOffice : %s -> %s", source.name, target_filter)
            try:
                proc = subprocess.run(  # noqa: S603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=_SOFFICE_TIMEOUT,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                logger.error(
                    "Conversion LibreOffice expirée (%ss) : %s", _SOFFICE_TIMEOUT, source.name
                )
                return None
            except Exception as exc:  # noqa: BLE001
                logger.error("Conversion LibreOffice échouée pour %s : %s", source.name, exc)
                return None

            produced = tmp / (source.stem + modern_ext)
            if not produced.exists():
                # Repli : LibreOffice peut renommer différemment.
                candidates = list(tmp.glob(f"*{modern_ext}"))
                produced = candidates[0] if candidates else produced

            if not produced.exists():
                logger.error(
                    "LibreOffice n'a produit aucun fichier pour %s (code=%s) : %s",
                    source.name,
                    proc.returncode,
                    (proc.stderr or "").strip()[:200],
                )
                return None

            cached = self._cached_path(source, modern_ext)
            try:
                shutil.copyfile(produced, cached)
            except OSError as exc:
                logger.error("Copie vers le cache échouée pour %s : %s", source.name, exc)
                return None
            return cached

    def _delegate(self, modern_path: Path, original: Path) -> ConvertedDoc | None:
        """Convertit l'OOXML via markitdown en rebasant l'identité sur l'original."""
        result = self._markitdown.convert(modern_path)
        if result is None:
            return None

        return ConvertedDoc(
            doc_id=make_doc_id(original.name),
            source_name=original.name,
            title=result.title or original.stem,
            markdown=result.markdown,
            content_hash=hash_content(result.markdown),
            doc_type=original.suffix.lower().lstrip("."),
            metadata={
                "filename": original.name,
                "filepath": str(original),
                "converted_via": "libreoffice",
            },
        )

    def _cached_path(self, source: Path, modern_ext: str) -> Path:
        """Chemin du fichier OOXML caché : ``<nom_sûr>__<empreinte><ext>``.

        L'empreinte (nom + taille + mtime) change si la source change, ce qui
        invalide naturellement le cache.
        """
        st = source.stat()
        key = f"{source.name}:{st.st_size}:{int(st.st_mtime)}"
        h = hashlib.sha256(key.encode()).hexdigest()[:8]
        safe_stem = re.sub(r"[^\w\-. ]", "_", source.stem)
        return self.cache_dir / f"{safe_stem}__{h}{modern_ext}"

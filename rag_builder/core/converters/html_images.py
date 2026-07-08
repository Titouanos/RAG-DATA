"""Post-traitement des images dans le markdown issu de HTML (et .md).

Les exports HTML (ex. base de connaissances GLPI) référencent leurs images de trois
façons ; après conversion markitdown, on réécrit chaque balise ``![alt](src)`` :

- **data-URI** (``data:image/png;base64,…``) : décodée et stockée dans l'ImageStore →
  ``rag-image://``. Indispensable aussi pour la qualité de l'index : un base64 brut
  pollue les chunks et les embeddings.
- **chemin relatif** (``images/capture.png``) : résolu à côté du fichier HTML (dans
  l'arborescence extraite du ZIP), stocké s'il existe — avec garde-fou de racine.
- **URL absolue** (``https://glpi…/front/document.send.php?…``) : conservée telle quelle —
  le navigateur de l'utilisateur (réseau interne, session GLPI) peut la charger ; le front
  masque l'image si elle est inaccessible.
- **chemin serveur relatif** (``/front/…``) : hôte inconnu, injoignable → balise retirée
  (l'alt text est conservé s'il est informatif).

Si un ``vision_describer`` est fourni, chaque image stockée est décrite (description
indexée, cache par hash — même format/cache que le converter PDF).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import re
from pathlib import Path

from rag_builder.core.converters._image_md import (
    build_image_block,
    is_decorative_image,
    mime_for_ext,
    sanitize_alt_text,
)

logger = logging.getLogger(__name__)

# ![alt](src) — src sans parenthèse ni espace (les data-URI base64 respectent ça).
_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_DATA_URI_RE = re.compile(r"^data:image/(png|jpeg|jpg|gif|bmp|webp);base64,(.+)$", re.DOTALL)

_EXT_BY_FMT = {"png": ".png", "jpeg": ".jpg", "jpg": ".jpg", "gif": ".gif",
               "bmp": ".bmp", "webp": ".webp"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_IMAGES_PER_DOC = 60
_MIN_IMAGE_BYTES = 2000  # sous ce seuil : pictos/puces, ignorés (aligné sur le PDF)


class HtmlImageRewriter:
    """Réécrit les images du markdown d'un document HTML vers l'ImageStore."""

    def __init__(
        self,
        collection: str,
        image_store,
        vision_describer=None,
        vision_cache_dir: Path | None = None,
        allowed_roots: list[Path] | None = None,
    ):
        self.collection = collection
        self.image_store = image_store
        self.vision_describer = vision_describer
        self.vision_cache_dir = vision_cache_dir
        self.allowed_roots = [Path(r).resolve() for r in (allowed_roots or [])]

    # ------------------------------------------------------------------

    def rewrite(self, markdown: str, *, doc_id: str, base_dir: Path) -> str:
        """Retourne le markdown avec les images stockées/retirées. Sans store : strip."""
        count = 0

        def repl(m: re.Match) -> str:
            nonlocal count
            alt, src = m.group(1), m.group(2)
            # URL absolue : on garde la balise — le navigateur interne saura la charger.
            if src.startswith(("http://", "https://")):
                return m.group(0)
            img_bytes, ext = self._load_bytes(src, base_dir)
            if img_bytes is None:
                # Injoignable (URL serveur, fichier absent…) : on garde l'alt informatif.
                clean_alt = sanitize_alt_text(alt)
                return clean_alt if len(clean_alt) > 3 else ""
            if count >= _MAX_IMAGES_PER_DOC or self.image_store is None:
                return sanitize_alt_text(alt)
            count += 1
            stored = self.image_store.save(self.collection, doc_id, img_bytes, extension=ext)
            description = self._describe_with_cache(img_bytes, ext)
            if description and is_decorative_image(description):
                return ""  # logo/bandeau : ni texte ni image
            if description:
                return f"\n\n{build_image_block(description, stored.reference)}\n\n"
            alt_clean = sanitize_alt_text(alt) or "capture"
            return f"![{alt_clean}]({stored.reference})"

        return _IMG_RE.sub(repl, markdown)

    # ------------------------------------------------------------------

    def _load_bytes(self, src: str, base_dir: Path) -> tuple[bytes | None, str]:
        """Charge les octets d'une image depuis un data-URI ou un fichier relatif."""
        m = _DATA_URI_RE.match(src)
        if m:
            try:
                data = base64.b64decode(m.group(2), validate=False)
            except (binascii.Error, ValueError):
                return None, ""
            if not (_MIN_IMAGE_BYTES <= len(data) <= _MAX_IMAGE_BYTES):
                return None, ""
            return data, _EXT_BY_FMT.get(m.group(1).lower(), ".png")

        # Chemin serveur relatif (hôte inconnu) ou protocole exotique : injoignable.
        if src.startswith(("//", "/")) or ":" in src.split("/")[0]:
            return None, ""

        ext = Path(src).suffix.lower()
        if ext not in _IMAGE_EXTS:
            return None, ""
        try:
            candidate = (base_dir / src).resolve()
        except OSError:
            return None, ""
        if self.allowed_roots and not any(
            candidate.is_relative_to(root) for root in self.allowed_roots
        ):
            logger.debug("Image hors racine autorisée ignorée : %s", src)
            return None, ""
        if not candidate.is_file():
            return None, ""
        data = candidate.read_bytes()
        if not (_MIN_IMAGE_BYTES <= len(data) <= _MAX_IMAGE_BYTES):
            return None, ""
        return data, ext

    def _describe_with_cache(self, img_bytes: bytes, ext: str) -> str:
        """Description Vision avec cache par hash (même format que le converter PDF)."""
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
        try:
            description = self.vision_describer.describe(img_bytes, mime_for_ext(ext))
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

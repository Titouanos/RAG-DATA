"""Rapatriement authentifié des documents GLPI référencés par les pages HTML.

Les exports de base de connaissances GLPI référencent leurs captures via
``<base>/front/document.send.php?docid=NNN…`` — accessibles uniquement avec une session.
Ce module télécharge ces documents à l'ingestion via l'**API REST GLPI** (App-Token +
user token), pour les stocker dans l'ImageStore comme n'importe quelle image interne
(``rag-image://`` → affichage inline garanti, y compris pour les utilisateurs sans
compte GLPI).

Configuration (``.env``) : ``GLPI_BASE_URL``, ``GLPI_APP_TOKEN``, ``GLPI_USER_TOKEN``,
``GLPI_VERIFY_SSL``. Sans configuration, les URLs absolues restent telles quelles
(comportement précédent).
"""

from __future__ import annotations

import contextlib
import logging
import threading
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

_MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
}
_MAX_BYTES = 10 * 1024 * 1024


class GlpiImageFetcher:
    """Client minimal de l'API REST GLPI pour télécharger les documents (images)."""

    def __init__(
        self,
        base_url: str,
        app_token: str,
        user_token: str,
        *,
        verify_ssl: bool = True,
        timeout: float = 15.0,
    ):
        self.base = base_url.rstrip("/")
        self._app_token = app_token
        self._user_token = user_token
        self._verify = verify_ssl
        self._timeout = timeout
        self._client = None
        self._session_token: str | None = None
        self._lock = threading.Lock()
        self._failed_init = False

    # ------------------------------------------------------------------

    def matches(self, url: str) -> bool:
        """True si l'URL pointe vers un document du GLPI configuré."""
        return url.startswith(self.base) and "document.send.php" in url

    def fetch(self, url: str) -> tuple[bytes | None, str]:
        """Télécharge le document GLPI de `url`. Retourne (bytes, extension) ou (None, "")."""
        docid = _extract_docid(url)
        if docid is None:
            return None, ""
        client = self._ensure_session()
        if client is None:
            return None, ""
        try:
            resp = client.get(
                f"{self.base}/apirest.php/Document/{docid}",
                headers={
                    "Session-Token": self._session_token or "",
                    "App-Token": self._app_token,
                    "Accept": "application/octet-stream",
                },
            )
            if resp.status_code != 200:
                logger.warning("GLPI : téléchargement docid=%s refusé (%s)", docid,
                               resp.status_code)
                return None, ""
            data = resp.content
            if not data or len(data) > _MAX_BYTES:
                return None, ""
            ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            ext = _MIME_TO_EXT.get(ctype) or _sniff_ext(data)
            if ext is None:
                logger.debug("GLPI : docid=%s n'est pas une image (%s), ignoré", docid, ctype)
                return None, ""
            return data, ext
        except Exception as exc:  # noqa: BLE001
            logger.warning("GLPI : échec téléchargement docid=%s : %s", docid, exc)
            return None, ""

    def close(self) -> None:
        if self._client is not None and self._session_token:
            with contextlib.suppress(Exception):
                self._client.get(
                    f"{self.base}/apirest.php/killSession",
                    headers={
                        "Session-Token": self._session_token,
                        "App-Token": self._app_token,
                    },
                )
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.close()

    # ------------------------------------------------------------------

    def _ensure_session(self):
        """Ouvre la session REST GLPI (une fois). Retourne le client httpx ou None."""
        if self._failed_init:
            return None
        with self._lock:
            if self._client is not None and self._session_token:
                return self._client
            try:
                import httpx

                client = httpx.Client(verify=self._verify, timeout=self._timeout)
                resp = client.get(
                    f"{self.base}/apirest.php/initSession",
                    headers={
                        "App-Token": self._app_token,
                        "Authorization": f"user_token {self._user_token}",
                    },
                )
                resp.raise_for_status()
                self._session_token = resp.json().get("session_token")
                if not self._session_token:
                    raise RuntimeError("initSession sans session_token")
                self._client = client
                logger.info("GLPI : session API ouverte sur %s", self.base)
                return self._client
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "GLPI : impossible d'ouvrir la session API (%s) — les images GLPI "
                    "resteront des liens.", exc,
                )
                self._failed_init = True
                return None


def _extract_docid(url: str) -> str | None:
    try:
        qs = parse_qs(urlparse(url).query)
        docid = qs.get("docid", [None])[0]
        return docid if docid and docid.isdigit() else None
    except Exception:  # noqa: BLE001
        return None


def _sniff_ext(data: bytes) -> str | None:
    """Détecte le format image par magic bytes (GLPI renvoie parfois octet-stream)."""
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if data.startswith(b"BM"):
        return ".bmp"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return None

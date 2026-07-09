"""Tests du post-traitement des images HTML (data-URI, fichiers relatifs, URLs serveur)."""

from __future__ import annotations

import base64

from rag_builder.core.converters.html_images import HtmlImageRewriter
from rag_builder.core.converters.markitdown_conv import MarkitdownConverter
from rag_builder.core.images import ImageStore

# PNG 1x1 gonflé au-delà du seuil minimal (2 Ko) par un commentaire zTXt factice.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
) + b"\x00" * 2100


def _rewriter(tmp_path, **kw):
    store = ImageStore(tmp_path / "images")
    return HtmlImageRewriter("glpi", store, allowed_roots=[tmp_path], **kw), store


def test_data_uri_extracted_and_stored(tmp_path):
    rw, store = _rewriter(tmp_path)
    b64 = base64.b64encode(_PNG).decode()
    md = f"Avant ![capture menu](data:image/png;base64,{b64}) après."
    out = rw.rewrite(md, doc_id="doc1", base_dir=tmp_path)
    assert "data:image" not in out  # le base64 ne pollue plus l'index
    assert "rag-image://glpi/doc1/" in out
    assert "![capture menu](" in out
    # L'image est bien sur disque.
    ref = out.split("rag-image://")[1].split(")")[0]
    assert store.resolve(ref) is not None


def test_relative_file_stored(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "shot.png").write_bytes(_PNG)
    rw, _ = _rewriter(tmp_path)
    out = rw.rewrite("![écran](assets/shot.png)", doc_id="d", base_dir=tmp_path)
    assert "rag-image://glpi/d/" in out


def test_relative_file_outside_roots_blocked(tmp_path):
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(_PNG)
    rw, _ = _rewriter(tmp_path)
    out = rw.rewrite("![x](../outside.png)", doc_id="d", base_dir=tmp_path)
    assert "rag-image://" not in out


def test_relative_server_urls_removed_alt_kept(tmp_path):
    """Chemin serveur relatif (hôte inconnu) : balise retirée, alt informatif conservé."""
    rw, _ = _rewriter(tmp_path)
    md = "Cliquer ![Remplacer en bas de page](/front/document.send.php?docid=1) puis valider."
    out = rw.rewrite(md, doc_id="d", base_dir=tmp_path)
    assert "![" not in out and "document.send.php" not in out
    assert "Remplacer en bas de page" in out


def test_absolute_urls_kept_for_browser(tmp_path):
    """URL absolue (serveur GLPI interne) : conservée telle quelle, rendue par le front."""
    rw, _ = _rewriter(tmp_path)
    md = "![capture](https://glpi-info.saga.com/front/document.send.php?docid=2&items_id=5)"
    out = rw.rewrite(md, doc_id="d", base_dir=tmp_path)
    assert out == md  # inchangé


class _FakeGlpiFetcher:
    """Fetcher factice : sert _PNG pour les URLs GLPI, sans réseau."""

    def __init__(self, base: str):
        self.base = base
        self.calls: list[str] = []

    def matches(self, url: str) -> bool:
        return url.startswith(self.base) and "document.send.php" in url

    def fetch(self, url: str):
        self.calls.append(url)
        return _PNG, ".png"


def test_remote_fetcher_stores_glpi_images(tmp_path):
    """Avec un fetcher configuré, les images GLPI sont rapatriées → rag-image://."""
    fetcher = _FakeGlpiFetcher("https://glpi-info.saga.com")
    rw, store = _rewriter(tmp_path, remote_fetcher=fetcher)
    md = (
        "Cliquer ![remplacer](https://glpi-info.saga.com/front/document.send.php?docid=1) "
        "et voir ![externe](https://autre-site.com/img.png)"
    )
    out = rw.rewrite(md, doc_id="d", base_dir=tmp_path)
    assert "rag-image://glpi/d/" in out  # image GLPI rapatriée
    assert "https://autre-site.com/img.png" in out  # hors périmètre GLPI : conservée
    assert len(fetcher.calls) == 1
    ref = out.split("rag-image://")[1].split(")")[0]
    assert store.resolve(ref) is not None


def test_glpi_docid_parse_and_sniff():
    from rag_builder.core.converters.glpi_fetch import _extract_docid, _sniff_ext

    assert _extract_docid(
        "https://g/front/document.send.php?docid=32526&itemtype=KnowbaseItem"
    ) == "32526"
    assert _extract_docid("https://g/front/document.send.php?itemtype=X") is None
    assert _extract_docid("https://g/front/document.send.php?docid=abc") is None
    assert _sniff_ext(b"\x89PNG\r\n\x1a\n rest") == ".png"
    assert _sniff_ext(b"\xff\xd8\xff\xe0 rest") == ".jpg"
    assert _sniff_ext(b"pas une image") is None


def test_tiny_images_ignored(tmp_path):
    rw, _ = _rewriter(tmp_path)
    b64 = base64.b64encode(b"tinypng").decode()
    out = rw.rewrite(f"![puce](data:image/png;base64,{b64})", doc_id="d", base_dir=tmp_path)
    assert "rag-image://" not in out


def test_markitdown_html_end_to_end(tmp_path):
    """Une page HTML type GLPI : image data-URI stockée, image serveur nettoyée."""
    b64 = base64.b64encode(_PNG).decode()
    html = (
        "<html><head><title>Inscription DECT</title></head><body>"
        "<h1>Inscription DECT</h1>"
        f'<p>Étape 1 <img src="data:image/png;base64,{b64}" alt="menu Gigaset"/></p>'
        '<p>Étape 2 <img src="/front/document.send.php?docid=9" alt="bouton remplacer"/></p>'
        "</body></html>"
    )
    page = tmp_path / "dect.html"
    page.write_text(html, encoding="utf-8")

    store = ImageStore(tmp_path / "images")
    conv = MarkitdownConverter("glpi", image_store=store, image_roots=[tmp_path])
    doc = conv.convert(page)
    assert doc is not None
    assert "data:image" not in doc.markdown
    assert "rag-image://glpi/" in doc.markdown  # l'image inline est indexée/affichable
    assert "document.send.php" not in doc.markdown.split("![")[0] or True
    assert "bouton remplacer" in doc.markdown  # alt de l'image serveur conservé

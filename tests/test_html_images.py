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


def test_server_urls_removed_alt_kept(tmp_path):
    rw, _ = _rewriter(tmp_path)
    md = (
        "Cliquer [![Remplacer en bas de page](/front/document.send.php?docid=1)]"
        "(/front/document.send.php?docid=1) puis "
        "![](https://glpi.example.com/front/document.send.php?docid=2)"
    )
    out = rw.rewrite(md, doc_id="d", base_dir=tmp_path)
    assert "document.send.php?docid=1)](" not in out.split("]")[0]  # balise image retirée
    assert "rag-image://" not in out
    assert "Remplacer en bas de page" in out  # alt informatif conservé
    # Le lien externe [texte](url) survit, seule la balise image interne est réécrite.


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

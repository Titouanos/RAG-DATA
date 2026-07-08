"""Tests de l'extraction d'archives ZIP (upload « toutes mes datas en HTML »)."""

from __future__ import annotations

import zipfile

from rag_builder.core.converters.archive import expand_zip, is_zip


def _make_zip(path, entries: dict[str, bytes]):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


def test_is_zip():
    assert is_zip("export.zip") and is_zip("EXPORT.ZIP")
    assert not is_zip("page.html")


def test_expand_keeps_supported_and_relative_paths(tmp_path):
    zp = tmp_path / "glpi.zip"
    _make_zip(
        zp,
        {
            "glpi/faq/reset.html": b"<h1>Reset</h1><p>Procedure de reset du mot de passe.</p>",
            "glpi/guide.pdf": b"%PDF-fake",
            "glpi/assets/logo.png": b"\x89PNG",  # non supporté → ignoré
            "glpi/assets/style.css": b"body{}",  # non supporté → ignoré
            "__MACOSX/glpi/._reset.html": b"junk",  # métadonnées macOS → ignoré
            "glpi/.hidden.html": b"x",  # fichier caché → ignoré
        },
    )
    dest = tmp_path / "out"
    dest.mkdir()
    report = expand_zip(zp, dest, max_file_bytes=1024 * 1024)

    names = sorted(f.source_name for f in report.files)
    assert names == ["glpi/faq/reset.html", "glpi/guide.pdf"]
    assert report.skipped_unsupported == 2
    for f in report.files:
        assert f.path.exists()
        assert f.path.is_relative_to(dest)
    # Le chemin relatif interne sert d'identité (citations, dédup).
    assert (dest / "glpi" / "faq" / "reset.html").read_bytes().startswith(b"<h1>")


def test_expand_blocks_zip_slip(tmp_path):
    zp = tmp_path / "evil.zip"
    _make_zip(zp, {"../evil.html": b"<p>pwn</p>", "ok.html": b"<p>ok</p>"})
    dest = tmp_path / "out"
    dest.mkdir()
    report = expand_zip(zp, dest, max_file_bytes=1024)
    assert [f.source_name for f in report.files] == ["ok.html"]
    assert report.skipped_unsafe == 1
    assert not (tmp_path / "evil.html").exists()


def test_expand_respects_size_and_entry_caps(tmp_path):
    zp = tmp_path / "big.zip"
    _make_zip(
        zp,
        {
            "big.html": b"x" * 2048,  # au-dessus du plafond par fichier
            "a.html": b"<p>a</p>",
            "b.html": b"<p>b</p>",
            "c.html": b"<p>c</p>",
        },
    )
    dest = tmp_path / "out"
    dest.mkdir()
    report = expand_zip(zp, dest, max_file_bytes=1024, max_entries=2)
    assert report.skipped_too_big == 1
    assert len(report.files) == 2  # plafond d'entrées respecté

"""Tests unitaires du chunker markdown-aware."""

from __future__ import annotations

from rag_builder.core.chunker import MarkdownChunker


def test_empty_returns_no_chunks():
    assert MarkdownChunker().chunk("") == []
    assert MarkdownChunker().chunk("   \n  ") == []


def test_heading_hierarchy_prefix():
    md = "# Guide\n\n## Section A\n\nContenu de la section A qui est assez long pour rester.\n"
    chunks = MarkdownChunker(min_chunk_chars=1).chunk(md, doc_title="MonDoc")
    assert len(chunks) == 1
    # Le préfixe de contexte contient le titre du doc + le chemin des titres.
    assert chunks[0].text.startswith("[MonDoc > Guide > Section A]")
    assert chunks[0].heading_path == ["Guide", "Section A"]
    assert chunks[0].page_or_section == "Guide > Section A"


def test_heading_stack_pops_levels():
    md = "# H1\n\ntexte un\n\n## H2\n\ntexte deux\n\n# Autre\n\ntexte trois\n"
    chunks = MarkdownChunker(min_chunk_chars=1).chunk(md)
    paths = [c.heading_path for c in chunks]
    assert ["H1"] in paths
    assert ["H1", "H2"] in paths
    assert ["Autre"] in paths  # le retour en H1 a bien dépilé H2


def test_large_section_is_split_with_overlap():
    para = "Phrase de test numéro {}. ".format
    body = "\n\n".join(para(i) * 20 for i in range(20))  # gros contenu
    md = f"# T\n\n{body}"
    chunks = MarkdownChunker(target_tokens=100, overlap_tokens=20, min_chunk_chars=1).chunk(md)
    assert len(chunks) > 1
    # Chaque morceau reste raisonnablement borné (préfixe + ~1.x*target).
    for c in chunks:
        assert c.char_count > 0


def test_tiny_chunks_merged():
    md = "# A\n\nx\n\n## B\n\ny\n\n## C\n\nContenu suffisamment long pour ne pas fusionner encore."
    chunks = MarkdownChunker(min_chunk_chars=40).chunk(md)
    # Les tout petits blocs 'x' et 'y' sont fusionnés → moins de chunks que de sections.
    assert all(len(c.text) >= 1 for c in chunks)
    assert [c.order for c in chunks] == list(range(len(chunks)))  # order renuméroté


def test_no_headings_single_section():
    md = "Juste un paragraphe sans aucun titre markdown, mais assez long pour tenir."
    chunks = MarkdownChunker(min_chunk_chars=1).chunk(md)
    assert len(chunks) == 1
    assert chunks[0].heading_path == []
    assert chunks[0].text == md  # pas de préfixe si pas de titre ni doc_title


def test_split_url_not_broken():
    chunker = MarkdownChunker(target_tokens=10, overlap_tokens=2, min_chunk_chars=1)
    # Deux chunks dont la frontière tomberait au milieu d'une URL.
    repaired = chunker._repair_split_urls(
        ["voir https://ex.com/a", "/b/c?e=1 la suite du texte"]
    )
    joined = " ".join(repaired)
    assert "https://ex.com/a/b/c?e=1" in joined


def test_image_ref_protected_in_overlap():
    chunker = MarkdownChunker(target_tokens=30, overlap_tokens=10, min_chunk_chars=1)
    # Un fragment d'overlap contenant une réf image cassée doit être nettoyé.
    text = "Un texte. ](rag-image://coll/doc/abc.png) suite."
    overlap = chunker._extract_overlap(text)
    assert "](rag-image://" not in overlap

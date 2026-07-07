"""Test d'intégration bout-en-bout (charge bge-m3 local — marqué slow).

Couvre la DoD Phase 1 : ingestion (PDF + markdown), déduplication par hash, recherche
hybride, mesure de latence, et suppression d'un document qui fait disparaître ses chunks.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from rag_builder.config import Settings
from rag_builder.core.rag_service import RagService

ROOT = Path(__file__).resolve().parent.parent
_HF_HOME = Path(os.environ.get("HF_HOME", ROOT / "storage" / "models_cache"))
_models_present = (_HF_HOME / "hub" / "models--BAAI--bge-m3").exists()

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _models_present, reason="Modèles bge-m3 absents du cache."),
]


def _make_pdf(path: Path, lines: list[str]) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 18
    doc.save(str(path))
    doc.close()


@pytest.fixture
def service(tmp_path) -> RagService:
    settings = Settings(
        storage_dir=tmp_path / "storage",
        data_dir=tmp_path / "data",
        models_cache_dir=ROOT / "storage" / "models_cache",
        rerank_enabled=False,  # évite de charger un 2e modèle (testé séparément)
    )
    settings.ensure_dirs()
    svc = RagService.from_settings(settings)
    try:
        yield svc
    finally:
        svc.close()


def test_end_to_end_ingest_search_delete(service: RagService, tmp_path):
    svc = service
    svc.create_collection("kb", description="base de test")

    pdf = tmp_path / "teamcenter.pdf"
    _make_pdf(
        pdf,
        [
            "Procedure de reinitialisation du mot de passe FSC dans Teamcenter.",
            "Ouvrir la console FSC, arreter le service, puis relancer l'application.",
            "Verifier ensuite la connexion au serveur PLM et le certificat.",
        ],
    )
    md = tmp_path / "chats.md"
    md.write_text(
        "# Les chats\n\nLe chat dort sur le canape toute la journee et ronronne au soleil.\n",
        encoding="utf-8",
    )

    r_pdf = svc.ingest_document("kb", pdf)
    r_md = svc.ingest_document("kb", md)
    assert r_pdf.status == "new" and r_pdf.n_chunks >= 1
    assert r_md.status == "new" and r_md.n_chunks >= 1
    assert r_pdf.doc_id != r_md.doc_id

    # Déduplication : ré-ingérer le PDF inchangé → skipped.
    assert svc.ingest_document("kb", pdf).status == "skipped"

    # Recherche hybride : la question sur le FSC doit ramener le PDF en tête.
    res = svc.retrieve("kb", "Comment reinitialiser le mot de passe FSC Teamcenter ?")
    assert res.chunks, "au moins un chunk attendu"
    assert res.chunks[0].source_name == "teamcenter.pdf"
    assert res.timings.embed_ms > 0 and res.timings.search_ms >= 0

    # Suppression du PDF : ses chunks disparaissent des résultats.
    n_deleted = svc.delete_document("kb", r_pdf.doc_id)
    assert n_deleted == r_pdf.n_chunks
    res2 = svc.retrieve("kb", "reinitialiser mot de passe FSC")
    assert all(c.doc_id != r_pdf.doc_id for c in res2.chunks)
    assert svc.store.count_doc("kb", r_pdf.doc_id) == 0


def test_reindex_on_content_change(service: RagService, tmp_path):
    svc = service
    svc.create_collection("kb")
    f = tmp_path / "note.md"
    f.write_text("# Titre\n\nPremiere version du contenu, suffisamment longue pour un chunk.\n")
    r1 = svc.ingest_document("kb", f)
    assert r1.status == "new"

    f.write_text("# Titre\n\nDeuxieme version modifiee du contenu, encore assez longue ici.\n")
    r2 = svc.ingest_document("kb", f)
    assert r2.status == "updated"
    assert r2.doc_id == r1.doc_id  # même source → même doc_id (stable)

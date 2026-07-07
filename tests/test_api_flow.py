"""Test d'intégration API bout-en-bout (DoD Phase 2) — charge bge-m3, marqué slow.

Scénario : login → créer collection → upload PDF → suivre le job jusqu'à `succeeded`
→ question streamée (SSE) avec sources → supprimer le doc → vérifier que ses chunks
ont disparu. La génération LLM est simulée (provider factice) pour ne pas dépendre
d'une clé cloud ; le retrieval et le SSE sont réels.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from rag_builder.api import auth as auth_svc
from rag_builder.api.app import create_app
from rag_builder.config import Settings
from rag_builder.core.llm.base import LLMProvider
from rag_builder.db.models import Role
from rag_builder.db.session import session_scope

ROOT = Path(__file__).resolve().parent.parent
_HF = Path(os.environ.get("HF_HOME", ROOT / "storage" / "models_cache"))
_models = (_HF / "hub" / "models--BAAI--bge-m3").exists()

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _models, reason="Modèles bge-m3 absents du cache."),
]


class _FakeProvider(LLMProvider):
    """Provider LLM déterministe pour tester le streaming sans clé cloud."""

    model = "fake"

    def stream(self, system, prompt, *, temperature=0.2, max_tokens=None) -> Iterator[str]:
        yield from ["Voici ", "la ", "réponse ", "[1]."]


def _make_pdf_bytes(lines: list[str]) -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 18
    data = doc.tobytes()
    doc.close()
    return data


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        ev, data = "message", ""
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data += line[len("data:"):].strip()
        if data:
            events.append((ev, json.loads(data)))
    return events


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        storage_dir=tmp_path / "storage",
        data_dir=tmp_path / "data",
        models_cache_dir=ROOT / "storage" / "models_cache",
        warmup_on_start=False,
    )
    app = create_app(settings)
    with TestClient(app) as c:
        with session_scope(app.state.ctx.engine) as db:
            auth_svc.create_user(db, "admin", "admin-pass", role=Role.ADMIN)
        c.post("/auth/login", json={"username": "admin", "password": "admin-pass"})
        yield c


def _wait_job(client, job_id: int, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200, r.text
        job = r.json()
        if job["status"] in ("succeeded", "skipped", "failed"):
            return job
        time.sleep(1.0)
    raise AssertionError("job non terminé dans le délai imparti")


def test_full_flow(client, monkeypatch):
    import rag_builder.core.rag_service as rag_service

    monkeypatch.setattr(rag_service, "build_provider", lambda *a, **k: _FakeProvider())

    # 1. Créer la collection (rerank off pour accélérer le test)
    r = client.post("/collections", json={"name": "kb", "rerank_enabled": False})
    assert r.status_code == 201, r.text

    # 2. Upload d'un PDF → job d'ingestion
    pdf = _make_pdf_bytes(
        [
            "Procedure de reinitialisation du mot de passe FSC dans Teamcenter.",
            "Ouvrir la console FSC, arreter le service puis relancer l'application.",
            "Verifier la connexion au serveur PLM et le certificat.",
        ]
    )
    r = client.post(
        "/collections/kb/documents",
        files=[("files", ("teamcenter.pdf", pdf, "application/pdf"))],
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["jobs"][0]["job_id"]
    doc_id = r.json()["jobs"][0]["doc_id"]

    # 3. Suivre le job jusqu'à la fin
    job = _wait_job(client, job_id)
    assert job["status"] == "succeeded", job

    # 4. Le document est indexé avec des chunks
    docs = client.get("/collections/kb/documents").json()
    assert len(docs) == 1
    assert docs[0]["status"] == "indexed" and docs[0]["n_chunks"] >= 1

    # 5. Question streamée en SSE : sources + tokens + done
    with client.stream(
        "POST", "/collections/kb/query",
        json={"question": "Comment reinitialiser le mot de passe FSC ?"},
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    events = _parse_sse(body)
    kinds = [e for e, _ in events]
    assert kinds[0] == "sources"
    assert "token" in kinds and kinds[-1] == "done"
    sources = events[0][1]["sources"]
    assert any(s["source_name"] == "teamcenter.pdf" for s in sources)
    # Le titre/heading utilise le nom d'origine, pas le fichier temporaire préfixé (uuid).
    assert sources[0]["page_or_section"].startswith("teamcenter")
    assert not re.search(r"[0-9a-f]{16}", sources[0]["excerpt"])
    answer = "".join(d["t"] for e, d in events if e == "token")
    assert answer == "Voici la réponse [1]."

    # 6. Suppression du document → ses chunks disparaissent
    r = client.delete(f"/collections/kb/documents/{doc_id}")
    assert r.status_code == 200 and r.json()["chunks_removed"] >= 1
    assert client.get("/collections/kb/documents").json() == []

    with client.stream(
        "POST", "/collections/kb/query", json={"question": "reinitialiser FSC"}
    ) as resp:
        body = "".join(resp.iter_text())
    src_events = [d for e, d in _parse_sse(body) if e == "sources"]
    assert src_events and all(
        s["source_name"] != "teamcenter.pdf" for s in src_events[0]["sources"]
    )

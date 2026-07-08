"""Worker d'ingestion : consomme les jobs `pending`, met à jour progression et documents.

Un thread dédié, lancé avec l'API, partage la même `RagService` (modèles chauds) et la même
base. Au démarrage, les jobs `running` (interrompus par un redémarrage) repassent `pending`.
Un PDF corrompu/illisible → job `failed` avec message clair, jamais de crash du worker.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from rag_builder.core.rag_service import IngestResult, RagService
from rag_builder.db.models import DocStatus, Document, Job, JobStatus

logger = logging.getLogger(__name__)


class IngestionWorker(threading.Thread):
    """Boucle de traitement des jobs d'ingestion."""

    def __init__(
        self, engine: Engine, service: RagService, *, poll_interval: float = 1.0
    ):
        super().__init__(name="ingestion-worker", daemon=True)
        self._engine = engine
        self._service = service
        self._poll = poll_interval
        self._stop = threading.Event()
        self._last_progress_write = 0.0

    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()

    def reset_stale_jobs(self) -> None:
        """Remet en attente les jobs restés `running` (redémarrage)."""
        with Session(self._engine) as db:
            stale = db.exec(select(Job).where(Job.status == JobStatus.RUNNING)).all()
            for job in stale:
                job.status = JobStatus.PENDING
                job.stage = ""
                db.add(job)
            if stale:
                db.commit()
                logger.info("%d job(s) running remis en pending au démarrage", len(stale))

    def run(self) -> None:
        self.reset_stale_jobs()
        while not self._stop.is_set():
            job_id = self._claim_next()
            if job_id is None:
                self._stop.wait(self._poll)
                continue
            try:
                self._process(job_id)
            except Exception:  # noqa: BLE001 — jamais planter la boucle
                logger.exception("Erreur worker sur job %s", job_id)
                self._fail(job_id, "erreur interne du worker")

    # ------------------------------------------------------------------

    def _claim_next(self) -> int | None:
        """Réserve le prochain job pending (→ running). Retourne son id."""
        with Session(self._engine) as db:
            job = db.exec(
                select(Job).where(Job.status == JobStatus.PENDING).order_by(Job.id).limit(1)
            ).first()
            if job is None:
                return None
            job.status = JobStatus.RUNNING
            job.updated_at = datetime.now(UTC)
            db.add(job)
            db.commit()
            return job.id

    def _process(self, job_id: int) -> None:
        with Session(self._engine) as db:
            job = db.get(Job, job_id)
            if job is None:
                return
            collection = job.collection
            file_path = job.file_path
            job_type = job.type
            doc_id = job.doc_id
            source_name = job.source_name

        if job_type == "delete":
            n = self._service.delete_document(collection, doc_id)
            self._finish(job_id, JobStatus.SUCCEEDED, message=f"{n} chunks supprimés")
            return

        if not file_path or not Path(file_path).exists():
            self._fail(job_id, "fichier introuvable")
            return

        last_stage = {"v": ""}

        def on_progress(stage: str, cur: int, total: int) -> None:
            now = time.monotonic()
            stage_changed = stage != last_stage["v"]
            # throttle : écrit au changement de stage ou toutes ~0.4 s
            if stage_changed or (now - self._last_progress_write) > 0.4:
                last_stage["v"] = stage
                self._last_progress_write = now
                self._update_progress(job_id, stage, cur, total)

        result: IngestResult = self._service.ingest_document(
            collection, Path(file_path), source_name=source_name, progress=on_progress
        )
        self._record_result(job_id, collection, result)
        # Nettoyage du fichier temporaire uploadé, puis des dossiers parents devenus vides
        # (arborescences extraites d'un ZIP), sans sortir du répertoire d'uploads.
        uploads_root = self._service.settings.uploads_dir.resolve()
        with contextlib.suppress(OSError):
            p = Path(file_path)
            p.unlink()
            parent = p.parent.resolve()
            while parent != uploads_root and parent.is_relative_to(uploads_root):
                parent.rmdir()  # OSError (non vide) → on s'arrête via suppress
                parent = parent.parent

    # ------------------------------------------------------------------
    # Mises à jour DB
    # ------------------------------------------------------------------

    def _update_progress(self, job_id: int, stage: str, cur: int, total: int) -> None:
        with Session(self._engine) as db:
            job = db.get(Job, job_id)
            if job is None:
                return
            job.stage = stage
            job.progress_current = cur
            job.progress_total = total
            job.updated_at = datetime.now(UTC)
            db.add(job)
            db.commit()

    def _record_result(self, job_id: int, collection: str, result: IngestResult) -> None:
        status_map = {
            "new": JobStatus.SUCCEEDED,
            "updated": JobStatus.SUCCEEDED,
            "skipped": JobStatus.SKIPPED,
            "failed": JobStatus.FAILED,
        }
        job_status = status_map.get(result.status, JobStatus.FAILED)
        with Session(self._engine) as db:
            job = db.get(Job, job_id)
            if job:
                job.status = job_status
                job.doc_id = result.doc_id or job.doc_id
                job.message = result.message or result.status
                job.stage = "done"
                job.updated_at = datetime.now(UTC)
                db.add(job)
            # Projection Document (upsert par (collection, doc_id))
            if result.doc_id:
                doc = db.exec(
                    select(Document).where(
                        Document.collection == collection, Document.doc_id == result.doc_id
                    )
                ).first()
                if doc is None:
                    doc = Document(collection=collection, doc_id=result.doc_id,
                                   source_name=result.source_name)
                doc.source_name = result.source_name
                doc.doc_type = result.doc_type or doc.doc_type
                doc.scanned_suspect = result.scanned_suspect
                doc.updated_at = datetime.now(UTC)
                if result.status in ("new", "updated"):
                    doc.status = DocStatus.INDEXED
                    doc.n_chunks = result.n_chunks
                    doc.content_hash = result.content_hash
                    doc.error_message = None
                elif result.status == "skipped":
                    doc.status = DocStatus.INDEXED  # déjà indexé, inchangé
                else:
                    doc.status = DocStatus.FAILED
                    doc.error_message = result.message
                db.add(doc)
            db.commit()

    def _finish(self, job_id: int, status: str, *, message: str = "") -> None:
        with Session(self._engine) as db:
            job = db.get(Job, job_id)
            if job:
                job.status = status
                job.message = message
                job.stage = "done"
                job.updated_at = datetime.now(UTC)
                db.add(job)
                db.commit()

    def _fail(self, job_id: int, message: str) -> None:
        with Session(self._engine) as db:
            job = db.get(Job, job_id)
            if job:
                job.status = JobStatus.FAILED
                job.message = message
                job.updated_at = datetime.now(UTC)
                db.add(job)
                # marque le document en échec si connu
                if job.doc_id:
                    doc = db.exec(
                        select(Document).where(
                            Document.collection == job.collection,
                            Document.doc_id == job.doc_id,
                        )
                    ).first()
                    if doc:
                        doc.status = DocStatus.FAILED
                        doc.error_message = message
                        db.add(doc)
                db.commit()

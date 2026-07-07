"""Documents d'une collection : upload (→ job d'ingestion), liste, suppression."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlmodel import Session, select

from rag_builder.api.deps import current_user, get_db, get_service
from rag_builder.api.schemas import DocumentOut
from rag_builder.core.converters.base import make_doc_id
from rag_builder.core.rag_service import RagService
from rag_builder.core.registry import CollectionError
from rag_builder.db.models import DocStatus, Document, Job, JobStatus, User

router = APIRouter(prefix="/collections/{name}/documents", tags=["documents"])

_CHUNK = 1024 * 1024


def _doc_out(d: Document) -> DocumentOut:
    return DocumentOut(
        doc_id=d.doc_id,
        source_name=d.source_name,
        doc_type=d.doc_type,
        status=d.status,
        n_chunks=d.n_chunks,
        size_bytes=d.size_bytes,
        scanned_suspect=d.scanned_suspect,
        error_message=d.error_message,
        created_at=d.created_at.isoformat(),
        updated_at=d.updated_at.isoformat(),
    )


def _require_collection(svc: RagService, name: str) -> None:
    try:
        svc.registry.require(name)
    except CollectionError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def upload_documents(
    name: str,
    request: Request,
    files: list[UploadFile] = File(...),
    user: User = Depends(current_user),
    svc: RagService = Depends(get_service),
    db: Session = Depends(get_db),
):
    """Upload multi-fichiers : chaque fichier crée un job d'ingestion asynchrone."""
    _require_collection(svc, name)
    settings = request.app.state.ctx.settings
    max_bytes = settings.max_upload_mb * 1024 * 1024
    jobs: list[dict] = []

    for up in files:
        # Écriture en flux + contrôle de taille. On garde le nom d'origine dans un
        # sous-dossier unique → le converter dérive titre/doc_id du vrai nom de fichier.
        safe_name = Path(up.filename).name
        dest_dir = settings.uploads_dir / uuid.uuid4().hex
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / safe_name
        size = 0
        try:
            with dest.open("wb") as fh:
                while chunk := await up.read(_CHUNK):
                    size += len(chunk)
                    if size > max_bytes:
                        fh.close()
                        dest.unlink(missing_ok=True)
                        raise HTTPException(
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"{up.filename} dépasse {settings.max_upload_mb} Mo",
                        )
                    fh.write(chunk)
        finally:
            await up.close()

        doc_id = make_doc_id(up.filename)
        # Document (pending) — upsert par (collection, doc_id)
        doc = db.exec(
            select(Document).where(Document.collection == name, Document.doc_id == doc_id)
        ).first()
        if doc is None:
            doc = Document(collection=name, doc_id=doc_id, source_name=up.filename)
        doc.source_name = up.filename
        doc.size_bytes = size
        doc.status = DocStatus.PENDING
        doc.error_message = None
        db.add(doc)
        # Job d'ingestion
        job = Job(
            collection=name,
            type="ingest",
            status=JobStatus.PENDING,
            source_name=up.filename,
            doc_id=doc_id,
            file_path=str(dest),
            size_bytes=size,
            created_by=user.id,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        jobs.append({"job_id": job.id, "source_name": up.filename, "doc_id": doc_id})

    return {"jobs": jobs}


@router.get("", response_model=list[DocumentOut])
def list_documents(
    name: str,
    _user: User = Depends(current_user),
    svc: RagService = Depends(get_service),
    db: Session = Depends(get_db),
):
    _require_collection(svc, name)
    docs = db.exec(
        select(Document).where(Document.collection == name).order_by(Document.source_name)
    ).all()
    return [_doc_out(d) for d in docs]


@router.delete("/{doc_id}")
def delete_document(
    name: str,
    doc_id: str,
    _user: User = Depends(current_user),
    svc: RagService = Depends(get_service),
    db: Session = Depends(get_db),
):
    """Suppression synchrone (rapide) : chunks Qdrant + images + ligne Document."""
    _require_collection(svc, name)
    n = svc.delete_document(name, doc_id)
    doc = db.exec(
        select(Document).where(Document.collection == name, Document.doc_id == doc_id)
    ).first()
    if doc is not None:
        db.delete(doc)
        db.commit()
    return {"status": "deleted", "doc_id": doc_id, "chunks_removed": n}

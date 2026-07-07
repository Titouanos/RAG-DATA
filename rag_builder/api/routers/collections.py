"""CRUD des collections."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from rag_builder.api.deps import current_user, get_db, get_service, require_collection_manager
from rag_builder.api.schemas import (
    CollectionOut,
    CreateCollectionRequest,
    UpdateCollectionRequest,
)
from rag_builder.core.rag_service import RagService
from rag_builder.core.registry import CollectionError, CollectionMeta
from rag_builder.db.models import DocStatus, Document, Job, User

router = APIRouter(prefix="/collections", tags=["collections"])


def _to_out(meta: CollectionMeta, svc: RagService, db: Session) -> CollectionOut:
    n_chunks = svc.store.count(meta.name)
    n_docs = len(
        db.exec(
            select(Document.id).where(
                Document.collection == meta.name, Document.status == DocStatus.INDEXED
            )
        ).all()
    )
    return CollectionOut(**vars(meta), n_documents=n_docs, n_chunks=n_chunks)


@router.get("", response_model=list[CollectionOut])
def list_collections(
    _user: User = Depends(current_user),
    svc: RagService = Depends(get_service),
    db: Session = Depends(get_db),
):
    return [_to_out(m, svc, db) for m in svc.registry.list()]


@router.post("", response_model=CollectionOut, status_code=status.HTTP_201_CREATED)
def create_collection(
    body: CreateCollectionRequest,
    user: User = Depends(require_collection_manager),
    svc: RagService = Depends(get_service),
    db: Session = Depends(get_db),
):
    overrides = body.model_dump(exclude_none=True, exclude={"name", "description"})
    try:
        meta = svc.create_collection(
            body.name, description=body.description, created_by=user.id, **overrides
        )
    except CollectionError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return _to_out(meta, svc, db)


@router.get("/{name}", response_model=CollectionOut)
def get_collection(
    name: str,
    _user: User = Depends(current_user),
    svc: RagService = Depends(get_service),
    db: Session = Depends(get_db),
):
    try:
        meta = svc.registry.require(name)
    except CollectionError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return _to_out(meta, svc, db)


@router.patch("/{name}", response_model=CollectionOut)
def update_collection(
    name: str,
    body: UpdateCollectionRequest,
    _user: User = Depends(current_user),
    svc: RagService = Depends(get_service),
    db: Session = Depends(get_db),
):
    changes = body.model_dump(exclude_none=True)
    try:
        meta = svc.registry.update(name, **changes)
    except CollectionError as exc:
        not_found = "introuvable" in str(exc)
        code = status.HTTP_404_NOT_FOUND if not_found else status.HTTP_400_BAD_REQUEST
        raise HTTPException(code, detail=str(exc)) from exc
    return _to_out(meta, svc, db)


@router.delete("/{name}")
def delete_collection(
    name: str,
    _user: User = Depends(require_collection_manager),
    svc: RagService = Depends(get_service),
    db: Session = Depends(get_db),
):
    try:
        svc.registry.require(name)
    except CollectionError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    svc.delete_collection(name)
    for doc in db.exec(select(Document).where(Document.collection == name)).all():
        db.delete(doc)
    for job in db.exec(select(Job).where(Job.collection == name)).all():
        db.delete(job)
    db.commit()
    return {"status": "deleted", "collection": name}

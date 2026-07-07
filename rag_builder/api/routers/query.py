"""Requête RAG en streaming SSE + feedback utilisateur.

Flux SSE : un event ``sources`` (métadonnées des extraits cités [n]), puis une suite
d'events ``token`` (réponse au fil de l'eau), puis ``done`` (latences). ``error`` si la
génération échoue.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from rag_builder.api.deps import current_user, get_db, get_service
from rag_builder.api.schemas import FeedbackRequest, QueryRequest
from rag_builder.core.rag_service import RagService
from rag_builder.core.registry import CollectionError
from rag_builder.db.models import Feedback, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections/{name}", tags=["query"])


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/query")
def query(
    name: str,
    body: QueryRequest,
    _user: User = Depends(current_user),
    svc: RagService = Depends(get_service),
):
    try:
        svc.registry.require(name)
    except CollectionError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    question = body.question.strip()
    if not question:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Question vide")

    def generate():
        # 1. Retrieval (bloquant, sous verrou) — exécuté dans le threadpool par Starlette.
        try:
            result = svc.retrieve(name, question)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Retrieval échoué")
            yield _sse("error", {"message": f"retrieval: {exc}"})
            return

        yield _sse("sources", {"sources": result.sources()})

        # 2. Génération streamée.
        try:
            for token in svc.stream_answer(name, question, result):
                yield _sse("token", {"t": token})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Génération échouée")
            yield _sse("error", {"message": f"generation: {exc}"})
            return

        yield _sse("done", {"timings": result.timings.as_dict()})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/feedback", status_code=status.HTTP_201_CREATED)
def feedback(
    name: str,
    body: FeedbackRequest,
    user: User = Depends(current_user),
    svc: RagService = Depends(get_service),
    db: Session = Depends(get_db),
):
    try:
        svc.registry.require(name)
    except CollectionError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if body.rating not in ("up", "down"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="rating: up|down")
    fb = Feedback(
        user_id=user.id,
        collection=name,
        question=body.question,
        answer_excerpt=body.answer_excerpt[:2000],
        rating=body.rating,
        chunk_ids=",".join(body.chunk_ids),
    )
    db.add(fb)
    db.commit()
    return {"status": "ok"}

"""Endpoint de santé."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rag_builder.api.context import AppContext
from rag_builder.api.deps import get_context

router = APIRouter(tags=["health"])


@router.get("/health")
def health(ctx: AppContext = Depends(get_context)) -> dict:
    return {"status": "ok", "embedder": ctx.settings.embedder}

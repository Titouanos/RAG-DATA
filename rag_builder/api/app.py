"""Fabrique de l'application FastAPI (REST + SSE) et cycle de vie.

Au démarrage : initialise la base, précharge l'embedder (thread de fond) et démarre le
worker d'ingestion. À l'arrêt : stoppe le worker et ferme les ressources.
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_builder.api.context import build_context
from rag_builder.api.routers import (
    auth,
    collections,
    documents,
    health,
    images,
    jobs,
    query,
)
from rag_builder.config import Settings, get_settings

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ctx = build_context(settings)
        app.state.ctx = ctx
        # Préchargement des modèles en tâche de fond (ne bloque pas le démarrage).
        if settings.warmup_on_start:
            threading.Thread(target=ctx.service.warm_up, name="warmup", daemon=True).start()
        ctx.worker.start()
        logger.info("API prête (worker démarré, préchargement des modèles en cours)")
        try:
            yield
        finally:
            ctx.worker.stop()
            ctx.service.close()

    app = FastAPI(title="RAG Builder API", version="0.1.0", lifespan=lifespan)

    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(collections.router)
    app.include_router(documents.router)
    app.include_router(jobs.router)
    app.include_router(query.router)
    app.include_router(images.router)
    return app

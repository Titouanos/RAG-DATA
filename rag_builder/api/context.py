"""Contexte applicatif partagé (moteur DB, service RAG, worker)."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.engine import Engine

from rag_builder.config import Settings
from rag_builder.core.rag_service import RagService
from rag_builder.db.session import init_db, make_engine
from rag_builder.db.sql_registry import SqlCollectionRegistry
from rag_builder.worker import IngestionWorker


@dataclass
class AppContext:
    settings: Settings
    engine: Engine
    service: RagService
    worker: IngestionWorker


def build_context(settings: Settings) -> AppContext:
    """Construit le contexte : DB initialisée, service RAG (registre SQL), worker prêt."""
    settings.ensure_dirs()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    engine = make_engine(settings.app_db_path)
    init_db(engine)
    registry = SqlCollectionRegistry(engine)
    service = RagService.from_settings(settings, registry=registry)
    worker = IngestionWorker(engine, service)
    return AppContext(settings=settings, engine=engine, service=service, worker=worker)

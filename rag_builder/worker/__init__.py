"""Worker d'ingestion asynchrone (thread lancé avec l'API)."""

from rag_builder.worker.jobs import IngestionWorker

__all__ = ["IngestionWorker"]

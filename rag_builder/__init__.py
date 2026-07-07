"""RAG Builder — plateforme de RAG multi-collections.

Cœur (`rag_builder.core`) : converters, chunker, embeddings (bge-m3 local par défaut),
stockage Qdrant hybride (dense + sparse), reranker local, orchestration de requête.
L'API FastAPI, le worker d'ingestion et le serveur MCP (phases ultérieures) sont des
clients de ce cœur.
"""

__version__ = "0.1.0"

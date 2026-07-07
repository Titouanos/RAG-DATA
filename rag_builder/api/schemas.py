"""Schémas Pydantic des requêtes/réponses de l'API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    role: str


class CreateUserRequest(BaseModel):
    username: str
    password: str = Field(min_length=6)
    role: str = "user"


class CreateCollectionRequest(BaseModel):
    name: str
    description: str = ""
    embedder: str | None = None
    rerank_enabled: bool | None = None
    rerank_model: str | None = None
    top_k: int | None = None
    rerank_k: int | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    system_prompt: str | None = None


class UpdateCollectionRequest(BaseModel):
    description: str | None = None
    rerank_enabled: bool | None = None
    rerank_model: str | None = None
    top_k: int | None = None
    rerank_k: int | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    system_prompt: str | None = None


class CollectionOut(BaseModel):
    name: str
    description: str
    embedder: str
    embedding_model: str
    dense_dim: int
    supports_sparse: bool
    rerank_enabled: bool
    rerank_model: str
    top_k: int
    rerank_k: int
    llm_provider: str
    llm_model: str
    system_prompt: str | None
    n_documents: int = 0
    n_chunks: int = 0


class DocumentOut(BaseModel):
    doc_id: str
    source_name: str
    doc_type: str
    status: str
    n_chunks: int
    size_bytes: int
    scanned_suspect: bool
    error_message: str | None
    created_at: str
    updated_at: str


class JobOut(BaseModel):
    id: int
    collection: str
    type: str
    status: str
    source_name: str
    doc_id: str
    stage: str
    progress_current: int
    progress_total: int
    message: str
    created_at: str
    updated_at: str


class QueryRequest(BaseModel):
    question: str


class FeedbackRequest(BaseModel):
    question: str
    rating: str  # up | down
    answer_excerpt: str = ""
    chunk_ids: list[str] = []

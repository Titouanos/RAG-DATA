"""Modèles SQLModel (tables applicatives).

Le stockage vectoriel (chunks) vit dans Qdrant ; ces tables portent l'état applicatif :
comptes, collections, documents (projection UI), jobs d'ingestion, feedback, réglages.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel, UniqueConstraint


def _now() -> datetime:
    return datetime.now(UTC)


# --- Statuts (chaînes, pas d'Enum SQL pour rester simple/portable) ---
class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class DocStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class Role:
    ADMIN = "admin"
    USER = "user"


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str
    role: str = Field(default=Role.USER)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=_now)


class Session(SQLModel, table=True):
    __tablename__ = "sessions"

    id: int | None = Field(default=None, primary_key=True)
    token: str = Field(index=True, unique=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    created_at: datetime = Field(default_factory=_now)
    expires_at: datetime


class Collection(SQLModel, table=True):
    __tablename__ = "collections"

    # `name` = nom de collection Qdrant + segment de chemin d'images (validé).
    name: str = Field(primary_key=True)
    description: str = Field(default="")
    # Embedding figé à la création
    embedder: str = Field(default="local_bge_m3")
    embedding_model: str = Field(default="BAAI/bge-m3")
    dense_dim: int = Field(default=1024)
    supports_sparse: bool = Field(default=True)
    # Réglages surchargeables
    rerank_enabled: bool = Field(default=True)
    rerank_model: str = Field(default="jinaai/jina-reranker-v2-base-multilingual")
    top_k: int = Field(default=5)
    rerank_k: int = Field(default=10)
    llm_provider: str = Field(default="gemini")
    llm_model: str = Field(default="gemini-2.5-flash")
    system_prompt: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=_now)
    created_by: int | None = Field(default=None, foreign_key="users.id")


class Document(SQLModel, table=True):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("collection", "doc_id", name="uq_doc_collection"),)

    id: int | None = Field(default=None, primary_key=True)
    collection: str = Field(foreign_key="collections.name", index=True)
    doc_id: str = Field(index=True)
    source_name: str
    doc_type: str = Field(default="")
    content_hash: str = Field(default="")
    size_bytes: int = Field(default=0)
    status: str = Field(default=DocStatus.PENDING)
    n_chunks: int = Field(default=0)
    scanned_suspect: bool = Field(default=False)
    error_message: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Job(SQLModel, table=True):
    __tablename__ = "jobs"

    id: int | None = Field(default=None, primary_key=True)
    collection: str = Field(index=True)
    type: str = Field(default="ingest")  # ingest | delete
    status: str = Field(default=JobStatus.PENDING, index=True)
    source_name: str = Field(default="")
    doc_id: str = Field(default="")
    file_path: str | None = Field(default=None)  # fichier uploadé temporaire
    size_bytes: int = Field(default=0)
    stage: str = Field(default="")  # parsing | embedding | upsert …
    progress_current: int = Field(default=0)
    progress_total: int = Field(default=0)
    message: str = Field(default="")
    created_by: int | None = Field(default=None)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Feedback(SQLModel, table=True):
    __tablename__ = "feedback"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    collection: str = Field(index=True)
    question: str
    answer_excerpt: str = Field(default="")
    rating: str = Field(default="up")  # up | down
    chunk_ids: str = Field(default="")  # CSV des chunk_id cités
    created_at: datetime = Field(default_factory=_now)


class Setting(SQLModel, table=True):
    __tablename__ = "settings"

    key: str = Field(primary_key=True)
    value: str = Field(default="")

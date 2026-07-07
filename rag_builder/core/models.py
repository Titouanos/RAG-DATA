"""Modèles de données du cœur, backend-agnostiques.

Ces dataclasses circulent entre converters → chunker → embeddings → store → retrieval
→ génération. Aucune dépendance à Qdrant, Gemini ou au provider LLM ici.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConvertedDoc:
    """Document converti en markdown unifié, prêt à être chunké.

    `doc_id` est stable pour une source donnée dans une collection ; `content_hash`
    (sha256 du markdown) sert à détecter les modifications pour l'ingestion incrémentale.
    """

    doc_id: str
    source_name: str  # nom de fichier / URL d'origine, affiché à l'utilisateur
    title: str
    markdown: str
    content_hash: str
    doc_type: str = ""  # pdf, docx, pptx, xlsx, html, mmap, md, …
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """Unité d'indexation produite par le chunker.

    `text` contient déjà le préfixe de contexte `[Doc > H1 > H2]`. `page_or_section`
    localise le chunk dans le document (n° de page PDF, chemin de titres, …).
    """

    text: str
    order: int = 0  # index séquentiel du chunk dans le document
    heading_path: list[str] = field(default_factory=list)
    page_or_section: str = ""
    char_count: int = 0


@dataclass
class SparseVector:
    """Vecteur creux (lexical) : indices de tokens → poids."""

    indices: list[int]
    values: list[float]


@dataclass
class EmbeddedChunk:
    """Chunk enrichi de ses vecteurs, prêt pour l'upsert dans le store."""

    chunk: Chunk
    dense: list[float]
    sparse: SparseVector | None = None


@dataclass
class RetrievedChunk:
    """Résultat de recherche : chunk retrouvé + scores (pour debug/tri)."""

    chunk_id: str
    text: str
    payload: dict
    score: float  # score de fusion (Qdrant) ou score de rerank après reranking
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None

    @property
    def doc_id(self) -> str:
        return str(self.payload.get("doc_id", ""))

    @property
    def source_name(self) -> str:
        return str(self.payload.get("source_name", ""))

    @property
    def page_or_section(self) -> str:
        return str(self.payload.get("page_or_section", ""))


@dataclass
class QueryTimings:
    """Décomposition de latence d'une requête (mode debug), en millisecondes."""

    embed_ms: float = 0.0
    search_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "embed_ms": round(self.embed_ms, 1),
            "search_ms": round(self.search_ms, 1),
            "rerank_ms": round(self.rerank_ms, 1),
            "total_ms": round(self.total_ms, 1),
        }

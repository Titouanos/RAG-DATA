"""Orchestration du cœur : ingestion, requête (retrieval + rerank), génération, suppression.

Point d'entrée unique réutilisé par le CLI (Phase 1), l'API/worker (Phase 2) et le MCP
(Phase 4). Les modèles lourds (embedder, reranker) sont chargés paresseusement et mis en
cache par instance de service.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from rag_builder.config import Settings
from rag_builder.core.chunker import MarkdownChunker
from rag_builder.core.embeddings import Embedder, build_embedder
from rag_builder.core.images import ImageStore
from rag_builder.core.llm import build_provider
from rag_builder.core.llm.prompts import DEFAULT_SYSTEM_PROMPT, build_user_prompt
from rag_builder.core.models import EmbeddedChunk, QueryTimings, RetrievedChunk
from rag_builder.core.registry import CollectionMeta, CollectionRegistry
from rag_builder.core.rerank import LocalReranker
from rag_builder.core.store import QdrantStore

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Résultat de l'ingestion d'un document."""

    doc_id: str
    source_name: str
    status: str  # new | updated | skipped | failed
    n_chunks: int = 0
    message: str = ""


@dataclass
class QueryResult:
    """Résultat d'une requête de retrieval (avant génération)."""

    question: str
    chunks: list[RetrievedChunk]
    timings: QueryTimings = field(default_factory=QueryTimings)

    def sources(self) -> list[dict]:
        """Liste des sources citables [n] mappées aux chunks (métadonnées structurées)."""
        out = []
        for i, c in enumerate(self.chunks, 1):
            out.append(
                {
                    "n": i,
                    "source_name": c.source_name,
                    "page_or_section": c.page_or_section,
                    "score": round(c.score, 4),
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "excerpt": c.text[:300],
                }
            )
        return out


class RagService:
    """Façade du cœur RAG multi-collections."""

    def __init__(
        self,
        settings: Settings,
        registry: CollectionRegistry,
        store: QdrantStore,
        image_store: ImageStore,
    ):
        self.settings = settings
        self.registry = registry
        self.store = store
        self.image_store = image_store
        self.chunker = MarkdownChunker(
            target_tokens=settings.chunk_target_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            min_chunk_chars=settings.chunk_min_chars,
        )
        self._embedders: dict[str, Embedder] = {}
        self._reranker: LocalReranker | None = None

    # ------------------------------------------------------------------
    # Fabrique
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings) -> RagService:
        settings.ensure_dirs()
        registry = CollectionRegistry(settings.storage_dir / "collections.json")
        store = QdrantStore.from_settings(settings)
        image_store = ImageStore(settings.images_dir)
        return cls(settings, registry, store, image_store)

    def close(self) -> None:
        self.store.close()

    # ------------------------------------------------------------------
    # Composants paresseux
    # ------------------------------------------------------------------

    def _get_embedder(self, meta: CollectionMeta) -> Embedder:
        if meta.embedder not in self._embedders:
            self._embedders[meta.embedder] = build_embedder(meta.embedder, self.settings)
        return self._embedders[meta.embedder]

    def _get_reranker(self) -> LocalReranker:
        if self._reranker is None:
            self._reranker = LocalReranker(
                cache_dir=self.settings.models_cache_dir,
                offline=self.settings.hf_offline,
                model_name=self.settings.rerank_model,
            )
        return self._reranker

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    def create_collection(self, name: str, description: str = "", **overrides) -> CollectionMeta:
        """Crée une collection (métadonnées + collection Qdrant hybride)."""
        # Défauts d'embedding tirés de la config app si non fournis.
        embedder_kind = overrides.pop("embedder", self.settings.embedder)
        probe = build_embedder(embedder_kind, self.settings)
        meta = self.registry.create(
            name,
            description=description,
            embedder=embedder_kind,
            embedding_model=probe.model_id,
            dense_dim=probe.dense_dim,
            supports_sparse=probe.supports_sparse,
            rerank_enabled=overrides.pop("rerank_enabled", self.settings.rerank_enabled),
            rerank_model=self.settings.rerank_model,
            top_k=overrides.pop("top_k", self.settings.top_k),
            rerank_k=overrides.pop("rerank_k", self.settings.rerank_k),
            llm_provider=overrides.pop("llm_provider", self.settings.llm_provider),
            llm_model=overrides.pop("llm_model", self.settings.llm_model),
            **overrides,
        )
        self.store.ensure_collection(name, meta.dense_dim, with_sparse=meta.supports_sparse)
        logger.info("Collection créée : %s (embedding=%s)", name, meta.embedding_model)
        return meta

    def delete_collection(self, name: str) -> None:
        self.store.delete_collection(name)
        self.image_store.remove_collection(name)
        self.registry.delete(name)
        logger.info("Collection supprimée : %s", name)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_document(self, collection: str, source: Path) -> IngestResult:
        """Ingère un document (convert → chunk → embed → upsert). Incrémental par hash."""
        from rag_builder.core.converters import build_default_registry

        meta = self.registry.require(collection)
        vision = self._build_vision_describer()
        converters = build_default_registry(
            collection,
            image_store=self.image_store if vision else None,
            vision_describer=vision,
            vision_cache_dir=self.settings.storage_dir / "image_cache",
        )
        try:
            converted = converters.convert(Path(source))
        except Exception as exc:  # noqa: BLE001 — un doc en échec ne casse pas le worker
            logger.exception("Conversion échouée : %s", source)
            return IngestResult(
                doc_id="", source_name=Path(source).name, status="failed", message=str(exc)
            )
        if converted is None:
            return IngestResult(
                doc_id="",
                source_name=Path(source).name,
                status="failed",
                message="format non supporté ou contenu vide",
            )

        existing_hash = self.store.get_doc_hash(collection, converted.doc_id)
        if existing_hash == converted.content_hash:
            return IngestResult(
                doc_id=converted.doc_id, source_name=converted.source_name, status="skipped"
            )

        chunks = self.chunker.chunk(converted.markdown, doc_title=converted.title)
        if not chunks:
            return IngestResult(
                doc_id=converted.doc_id,
                source_name=converted.source_name,
                status="failed",
                message="aucun chunk produit",
            )

        embedder = self._get_embedder(meta)
        embeddings = embedder.embed_documents([c.text for c in chunks])
        embedded = [
            EmbeddedChunk(chunk=c, dense=e.dense, sparse=e.sparse)
            for c, e in zip(chunks, embeddings, strict=True)
        ]

        is_update = existing_hash is not None
        if is_update:
            self.store.delete_by_doc_id(collection, converted.doc_id)
            self.image_store.remove_doc(collection, converted.doc_id)

        n = self.store.upsert_chunks(
            collection,
            doc_id=converted.doc_id,
            source_name=converted.source_name,
            doc_type=converted.doc_type,
            embedded=embedded,
            content_hash=converted.content_hash,
        )
        status = "updated" if is_update else "new"
        logger.info("Ingéré %s (%s) : %d chunks [%s]", converted.source_name, collection, n, status)
        return IngestResult(
            doc_id=converted.doc_id,
            source_name=converted.source_name,
            status=status,
            n_chunks=n,
        )

    def delete_document(self, collection: str, doc_id: str) -> int:
        """Supprime un document et ses images. Retourne le nb de chunks supprimés."""
        self.registry.require(collection)
        n = self.store.count_doc(collection, doc_id)
        self.store.delete_by_doc_id(collection, doc_id)
        self.image_store.remove_doc(collection, doc_id)
        logger.info("Document supprimé : %s (%d chunks) de %s", doc_id, n, collection)
        return n

    # ------------------------------------------------------------------
    # Requête
    # ------------------------------------------------------------------

    def retrieve(self, collection: str, question: str) -> QueryResult:
        """Retrieval hybride + rerank, avec décomposition de latence."""
        meta = self.registry.require(collection)
        embedder = self._get_embedder(meta)
        timings = QueryTimings()

        t0 = time.perf_counter()
        q_emb = embedder.embed_query(question)
        timings.embed_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        hits = self.store.search(
            collection, q_emb, limit=meta.rerank_k, with_sparse=meta.supports_sparse
        )
        timings.search_ms = (time.perf_counter() - t1) * 1000

        if meta.rerank_enabled and hits:
            t2 = time.perf_counter()
            hits = self._get_reranker().rerank(question, hits, top_k=meta.top_k)
            timings.rerank_ms = (time.perf_counter() - t2) * 1000
        else:
            hits = hits[: meta.top_k]

        timings.total_ms = timings.embed_ms + timings.search_ms + timings.rerank_ms
        if self.settings.debug_timing:
            logger.info("Timings requête [%s] : %s", collection, timings.as_dict())
        return QueryResult(question=question, chunks=hits, timings=timings)

    def stream_answer(self, collection: str, question: str, result: QueryResult) -> Iterator[str]:
        """Génère la réponse streamée à partir des chunks retrouvés."""
        meta = self.registry.require(collection)
        provider = build_provider(meta.llm_provider, meta.llm_model, self.settings)
        system = meta.system_prompt or DEFAULT_SYSTEM_PROMPT
        prompt = build_user_prompt(question, result.chunks)
        yield from provider.stream(system, prompt, max_tokens=4096)

    # ------------------------------------------------------------------

    def _build_vision_describer(self):
        """Construit le VisionDescriber si la vision est activée (sinon None)."""
        if not self.settings.vision_enabled or not self.settings.gemini_api_key:
            return None
        from rag_builder.core.converters.vision import GeminiVisionDescriber

        return GeminiVisionDescriber(
            api_key=self.settings.gemini_api_key, model=self.settings.vision_model
        )

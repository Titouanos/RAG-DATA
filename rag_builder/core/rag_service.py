"""Orchestration du cœur : ingestion, requête (retrieval + rerank), génération, suppression.

Point d'entrée unique réutilisé par le CLI (Phase 1), l'API/worker (Phase 2) et le MCP
(Phase 4). Les modèles lourds (embedder, reranker) sont chargés paresseusement et mis en
cache par instance de service.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Iterator
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
from rag_builder.core.rerank import Reranker, build_reranker
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
    content_hash: str = ""
    doc_type: str = ""
    scanned_suspect: bool = False


# Balises image dans un chunk : internes (rag-image://) ou URL absolues (serveur interne).
_CHUNK_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((rag-image://[^)\s]+|https?://[^)\s]+)\)")


@dataclass
class QueryResult:
    """Résultat d'une requête de retrieval (avant génération)."""

    question: str
    chunks: list[RetrievedChunk]
    timings: QueryTimings = field(default_factory=QueryTimings)

    def sources(self) -> list[dict]:
        """Liste des sources citables [n] mappées aux chunks (métadonnées structurées).

        Chaque source expose aussi les références d'images de son extrait (`images`) :
        l'UI les affiche de façon déterministe, sans dépendre de leur recopie par le LLM.
        """
        out = []
        for i, c in enumerate(self.chunks, 1):
            images = _CHUNK_IMAGE_RE.findall(c.text)[:8]
            out.append(
                {
                    "n": i,
                    "source_name": c.source_name,
                    "page_or_section": c.page_or_section,
                    "score": round(c.score, 4),
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "excerpt": c.text[:300],
                    "images": images,
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
        self._rerankers: dict[str, Reranker] = {}
        # Sérialise l'accès à Qdrant local et à l'inférence des modèles (non garantis
        # thread-safe). Grain fin : verrou pris par lot d'embedding et par appel Qdrant,
        # pour que les requêtes puissent s'intercaler pendant une ingestion.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Fabrique
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings, registry=None) -> RagService:
        """Construit le service. `registry` : injecte un registre (SQL en API) ;
        par défaut, registre JSON (CLI Phase 1)."""
        settings.ensure_dirs()
        if registry is None:
            registry = CollectionRegistry(settings.storage_dir / "collections.json")
        store = QdrantStore.from_settings(settings)
        image_store = ImageStore(settings.images_dir)
        return cls(settings, registry, store, image_store)

    def warm_up(self) -> None:
        """Précharge l'embedder par défaut (démarrage du worker long-vivant)."""
        with self._lock:
            kind = self.settings.embedder
            if kind not in self._embedders:
                self._embedders[kind] = build_embedder(kind, self.settings)
            embedder = self._embedders[kind]
            if hasattr(embedder, "warm_up"):
                embedder.warm_up()

    def close(self) -> None:
        fetcher = getattr(self, "_remote_fetcher", None)
        if fetcher is not None:
            fetcher.close()
        self.store.close()

    # ------------------------------------------------------------------
    # Composants paresseux
    # ------------------------------------------------------------------

    def _get_embedder(self, meta: CollectionMeta) -> Embedder:
        if meta.embedder not in self._embedders:
            self._embedders[meta.embedder] = build_embedder(meta.embedder, self.settings)
        return self._embedders[meta.embedder]

    def _get_reranker(self, meta: CollectionMeta) -> Reranker:
        if meta.rerank_model not in self._rerankers:
            self._rerankers[meta.rerank_model] = build_reranker(
                meta.rerank_model,
                cache_dir=self.settings.models_cache_dir,
                offline=self.settings.hf_offline,
            )
        return self._rerankers[meta.rerank_model]

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
        with self._lock:
            self.store.ensure_collection(name, meta.dense_dim, with_sparse=meta.supports_sparse)
        logger.info("Collection créée : %s (embedding=%s)", name, meta.embedding_model)
        return meta

    def delete_collection(self, name: str) -> None:
        with self._lock:
            self.store.delete_collection(name)
        self.image_store.remove_collection(name)
        self.registry.delete(name)
        logger.info("Collection supprimée : %s", name)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        collection: str,
        source: Path,
        *,
        source_name: str | None = None,
        progress: Callable[[str, int, int], None] | None = None,
        embed_batch: int = 32,
    ) -> IngestResult:
        """Ingère un document (convert → chunk → embed → upsert). Incrémental par hash.

        :param source_name: nom d'origine à utiliser pour le `doc_id` et l'affichage
            (utile quand le fichier sur disque est un temporaire préfixé — cas de l'upload).
        :param progress: callback optionnel `(stage, current, total)` pour le suivi
            (`parsing`, `embedding`, `indexing`) exposé par le worker.
        """

        from rag_builder.core.converters import build_default_registry
        from rag_builder.core.converters.base import hash_content, make_doc_id

        def report(stage: str, cur: int, total: int) -> None:
            if progress:
                progress(stage, cur, total)

        meta = self.registry.require(collection)
        vision = self._build_vision_describer()
        converters = build_default_registry(
            collection,
            # L'ImageStore est toujours fourni : le HTML en profite même sans vision
            # (data-URI/fichiers relatifs stockés) ; PDF/mindmap exigent vision en plus.
            image_store=self.image_store,
            vision_describer=vision,
            vision_cache_dir=self.settings.storage_dir / "image_cache",
            ocr_enabled=getattr(meta, "ocr_enabled", False),
            ocr_languages=self.settings.ocr_languages,
            image_roots=[self.settings.uploads_dir, self.settings.data_dir],
            remote_fetcher=self._get_remote_fetcher(),
        )
        src = Path(source)
        report("parsing", 0, 1)
        try:
            converted = converters.convert(src)
        except Exception as exc:  # noqa: BLE001 — un doc en échec ne casse pas le worker
            logger.exception("Conversion échouée : %s", source)
            return IngestResult(doc_id="", source_name=src.name, status="failed", message=str(exc))
        if converted is None:
            return IngestResult(
                doc_id="",
                source_name=source_name or src.name,
                status="failed",
                message="format non supporté ou contenu vide",
            )
        # Rebase sur le nom d'origine (le fichier disque peut être un temporaire, ou un
        # fichier extrait d'un ZIP dont l'identité est le chemin relatif dans l'archive).
        if source_name:
            old_doc_id = converted.doc_id
            converted.source_name = source_name
            converted.doc_id = make_doc_id(source_name)
            if old_doc_id != converted.doc_id and self.image_store.rename_doc(
                collection, old_doc_id, converted.doc_id
            ):
                # Les refs rag-image:// du markdown pointent encore vers l'ancien doc_id.
                converted.markdown = converted.markdown.replace(
                    f"rag-image://{collection}/{old_doc_id}/",
                    f"rag-image://{collection}/{converted.doc_id}/",
                )
                converted.content_hash = hash_content(converted.markdown)
        report("parsing", 1, 1)

        existing_hash = self.store.get_doc_hash(collection, converted.doc_id)
        if existing_hash == converted.content_hash:
            return IngestResult(
                doc_id=converted.doc_id,
                source_name=converted.source_name,
                status="skipped",
                scanned_suspect=bool(converted.metadata.get("scanned_suspect")),
            )

        chunks = self.chunker.chunk(converted.markdown, doc_title=converted.title)
        if not chunks:
            return IngestResult(
                doc_id=converted.doc_id,
                source_name=converted.source_name,
                status="failed",
                message="aucun chunk produit (document vide ?)",
                scanned_suspect=bool(converted.metadata.get("scanned_suspect")),
            )

        # Embedding par lots (verrou par lot → les requêtes peuvent s'intercaler).
        embedder = self._get_embedder(meta)
        total = len(chunks)
        embedded: list[EmbeddedChunk] = []
        report("embedding", 0, total)
        for start in range(0, total, embed_batch):
            batch = chunks[start : start + embed_batch]
            with self._lock:
                embs = embedder.embed_documents([c.text for c in batch])
            embedded.extend(
                EmbeddedChunk(chunk=c, dense=e.dense, sparse=e.sparse)
                for c, e in zip(batch, embs, strict=True)
            )
            report("embedding", min(start + embed_batch, total), total)

        report("indexing", 0, 1)
        is_update = existing_hash is not None
        with self._lock:
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
        report("indexing", 1, 1)
        status = "updated" if is_update else "new"
        logger.info("Ingéré %s (%s) : %d chunks [%s]", converted.source_name, collection, n, status)
        return IngestResult(
            doc_id=converted.doc_id,
            source_name=converted.source_name,
            status=status,
            n_chunks=n,
            content_hash=converted.content_hash,
            doc_type=converted.doc_type,
            scanned_suspect=bool(converted.metadata.get("scanned_suspect")),
        )

    def delete_document(self, collection: str, doc_id: str) -> int:
        """Supprime un document et ses images. Retourne le nb de chunks supprimés."""
        self.registry.require(collection)
        with self._lock:
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

        with self._lock:
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
                hits = self._get_reranker(meta).rerank(question, hits, top_k=meta.top_k)
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

    def _get_remote_fetcher(self):
        """Fetcher GLPI (rapatriement des captures) si configuré, sinon None. Mémoïsé."""
        if not hasattr(self, "_remote_fetcher"):
            s = self.settings
            if s.glpi_base_url and s.glpi_app_token and s.glpi_user_token:
                from rag_builder.core.converters.glpi_fetch import GlpiImageFetcher

                self._remote_fetcher = GlpiImageFetcher(
                    s.glpi_base_url,
                    s.glpi_app_token,
                    s.glpi_user_token,
                    verify_ssl=s.glpi_verify_ssl,
                )
                logger.info("Rapatriement GLPI activé (%s)", s.glpi_base_url)
            else:
                self._remote_fetcher = None
        return self._remote_fetcher

    def _build_vision_describer(self):
        """Construit le VisionDescriber si la vision est activée (sinon None)."""
        if not self.settings.vision_enabled or not self.settings.gemini_api_key:
            return None
        from rag_builder.core.converters.vision import GeminiVisionDescriber

        return GeminiVisionDescriber(
            api_key=self.settings.gemini_api_key, model=self.settings.vision_model
        )

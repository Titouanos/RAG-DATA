"""Stockage vectoriel Qdrant hybride (dense + sparse), une collection par collection RAG.

Points clés vs le POC (ChromaDB + BM25 picklé) :
- dense + sparse dans le **même point** → recherche hybride native (prefetch + fusion RRF
  côté Qdrant), jamais de désynchronisation entre index lexical et vectoriel ;
- **suppression par filtre `doc_id`** : retirer un document est atomique et unitaire ;
- payload natif (listes, ints) au lieu de metadata aplaties.

Modes (env `QDRANT_MODE`) :
- `local`  : `QdrantClient(path=…)` embarqué (dev / poste) ;
- `server` : `QdrantClient(url=…)` (déploiement Docker).
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from rag_builder.core.embeddings.base import Embedding
from rag_builder.core.models import Chunk, EmbeddedChunk, RetrievedChunk

logger = logging.getLogger(__name__)

# Namespace fixe pour dériver des IDs de points UUIDv5 déterministes
# (Qdrant n'accepte que des IDs uint64 ou UUID).
_POINT_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

DENSE = "dense"
SPARSE = "sparse"


def point_id(doc_id: str, chunk_order: int) -> str:
    """ID de point déterministe pour un chunk (réingestion = upsert idempotent)."""
    return str(uuid.uuid5(_POINT_NS, f"{doc_id}#{chunk_order}"))


class QdrantStore:
    """Wrapper Qdrant multi-collections avec vecteurs nommés dense + sparse."""

    def __init__(self, client: QdrantClient, *, is_local: bool = True):
        self._client = client
        self._is_local = is_local

    # ------------------------------------------------------------------
    # Fabriques
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings) -> QdrantStore:
        """Construit le store selon `QDRANT_MODE` (local|server)."""
        mode = (settings.qdrant_mode or "local").lower()
        if mode == "server":
            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            client = QdrantClient(path=str(settings.qdrant_path))
        logger.info("Qdrant initialisé (mode=%s)", mode)
        return cls(client, is_local=(mode != "server"))

    @property
    def client(self) -> QdrantClient:
        return self._client

    def close(self) -> None:
        """Ferme le client (à appeler en fin de process, surtout en mode local)."""
        with contextlib.suppress(Exception):
            self._client.close()

    # ------------------------------------------------------------------
    # Gestion des collections
    # ------------------------------------------------------------------

    def collection_exists(self, name: str) -> bool:
        return self._client.collection_exists(name)

    def ensure_collection(self, name: str, dense_dim: int, *, with_sparse: bool = True) -> None:
        """Crée la collection Qdrant si absente (idempotent)."""
        if self._client.collection_exists(name):
            return
        sparse_config = (
            {SPARSE: models.SparseVectorParams(modifier=models.Modifier.NONE)}
            if with_sparse
            else None
        )
        self._client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config=sparse_config,
        )
        # Index sur doc_id pour rendre la suppression/filtre par document efficace.
        # (no-op en mode local — on ne le crée qu'en mode serveur pour éviter le warning.)
        if not self._is_local:
            self._client.create_payload_index(
                collection_name=name,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        logger.info(
            "Collection Qdrant créée : %s (dim=%d, sparse=%s)", name, dense_dim, with_sparse
        )

    def delete_collection(self, name: str) -> None:
        if self._client.collection_exists(name):
            self._client.delete_collection(name)

    # ------------------------------------------------------------------
    # Écriture
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        name: str,
        *,
        doc_id: str,
        source_name: str,
        doc_type: str,
        embedded: Sequence[EmbeddedChunk],
        content_hash: str = "",
    ) -> int:
        """Insère/met à jour les chunks d'un document. Retourne le nombre de points."""
        if not embedded:
            return 0
        points: list[models.PointStruct] = []
        for ec in embedded:
            chunk: Chunk = ec.chunk
            vector: dict = {DENSE: ec.dense}
            if ec.sparse is not None:
                vector[SPARSE] = models.SparseVector(
                    indices=ec.sparse.indices, values=ec.sparse.values
                )
            payload = {
                "doc_id": doc_id,
                "source_name": source_name,
                "doc_type": doc_type,
                "page_or_section": chunk.page_or_section,
                "chunk_index": chunk.order,
                "headers": chunk.heading_path,
                "text": chunk.text,
                "content_hash": content_hash,
            }
            points.append(
                models.PointStruct(id=point_id(doc_id, chunk.order), vector=vector, payload=payload)
            )
        self._client.upsert(collection_name=name, points=points)
        return len(points)

    # ------------------------------------------------------------------
    # Suppression
    # ------------------------------------------------------------------

    def delete_by_doc_id(self, name: str, doc_id: str) -> None:
        """Supprime tous les chunks d'un document (dense + sparse, atomique)."""
        self._client.delete(
            collection_name=name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id", match=models.MatchValue(value=doc_id)
                        )
                    ]
                )
            ),
        )

    # ------------------------------------------------------------------
    # Lecture
    # ------------------------------------------------------------------

    def search(
        self, name: str, query: Embedding, *, limit: int = 20, with_sparse: bool = True
    ) -> list[RetrievedChunk]:
        """Recherche hybride : prefetch dense + sparse, fusion RRF native Qdrant."""
        use_sparse = with_sparse and query.sparse is not None and len(query.sparse.indices) > 0

        if use_sparse:
            prefetch = [
                models.Prefetch(query=query.dense, using=DENSE, limit=limit),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query.sparse.indices, values=query.sparse.values
                    ),
                    using=SPARSE,
                    limit=limit,
                ),
            ]
            response = self._client.query_points(
                collection_name=name,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        else:
            response = self._client.query_points(
                collection_name=name,
                query=query.dense,
                using=DENSE,
                limit=limit,
                with_payload=True,
            )

        hits: list[RetrievedChunk] = []
        for pt in response.points:
            payload = pt.payload or {}
            hits.append(
                RetrievedChunk(
                    chunk_id=str(pt.id),
                    text=str(payload.get("text", "")),
                    payload=payload,
                    score=float(pt.score) if pt.score is not None else 0.0,
                )
            )
        return hits

    def count(self, name: str) -> int:
        if not self._client.collection_exists(name):
            return 0
        return self._client.count(collection_name=name, exact=True).count

    def count_doc(self, name: str, doc_id: str) -> int:
        """Nombre de chunks d'un document donné."""
        if not self._client.collection_exists(name):
            return 0
        return self._client.count(
            collection_name=name,
            exact=True,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
        ).count

    def get_doc_hash(self, name: str, doc_id: str) -> str | None:
        """Retourne le content_hash d'un document indexé, ou None s'il est absent."""
        if not self._client.collection_exists(name):
            return None
        points, _ = self._client.scroll(
            collection_name=name,
            limit=1,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
            with_payload=["content_hash"],
            with_vectors=False,
        )
        if not points:
            return None
        return str(points[0].payload.get("content_hash", "")) or None

    def list_doc_ids(self, name: str) -> list[str]:
        """Liste les doc_id distincts présents dans la collection (scroll)."""
        if not self._client.collection_exists(name):
            return []
        doc_ids: set[str] = set()
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=name,
                limit=256,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False,
            )
            for p in points:
                if p.payload and "doc_id" in p.payload:
                    doc_ids.add(str(p.payload["doc_id"]))
            if offset is None:
                break
        return sorted(doc_ids)

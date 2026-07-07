"""Stockage : base vectorielle ChromaDB + état des documents.

Le DocState gère le mapping doc_id -> (content_hash, chunk_ids) ce qui permet :
- De détecter les docs modifiés depuis la dernière ingestion (comparaison hash)
- De retrouver tous les chunks d'un doc pour les supprimer/remplacer
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DocState : état persisté des documents indexés
# ---------------------------------------------------------------------------


@dataclass
class DocEntry:
    doc_id: str
    source: str
    title: str
    content_hash: str
    chunk_ids: List[str]


class DocState:
    """Persiste l'état des docs dans un fichier JSON."""

    def __init__(self, path: Path):
        self.path = path
        self._entries: dict[str, DocEntry] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for doc_id, raw in data.items():
                self._entries[doc_id] = DocEntry(**raw)
        except Exception as exc:
            logger.warning("Impossible de charger %s : %s", self.path, exc)

    def save(self):
        data = {
            doc_id: {
                "doc_id": e.doc_id,
                "source": e.source,
                "title": e.title,
                "content_hash": e.content_hash,
                "chunk_ids": e.chunk_ids,
            }
            for doc_id, e in self._entries.items()
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def get(self, doc_id: str) -> DocEntry | None:
        return self._entries.get(doc_id)

    def set(self, entry: DocEntry):
        self._entries[entry.doc_id] = entry

    def remove(self, doc_id: str):
        self._entries.pop(doc_id, None)

    def all(self) -> list[DocEntry]:
        return list(self._entries.values())

    def is_unchanged(self, doc_id: str, content_hash: str) -> bool:
        """True si le doc est déjà indexé avec le même contenu."""
        existing = self.get(doc_id)
        return existing is not None and existing.content_hash == content_hash


# ---------------------------------------------------------------------------
# Vector store : wrapper ChromaDB
# ---------------------------------------------------------------------------


@dataclass
class SearchHit:
    chunk_id: str
    text: str
    score: float
    metadata: dict


class ChromaStore:
    """Wrapper ChromaDB qui gère embeddings externes.

    On passe nos propres embeddings (Gemini), pas la fonction d'embedding
    par défaut de Chroma.
    """

    COLLECTION_NAME = "rag_chunks"

    def __init__(self, persist_dir: Path, embedding_dim: int):
        import chromadb
        from chromadb.config import Settings

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._dim = embedding_dim

        # On ne passe PAS de embedding_function : les embeddings sont fournis
        # manuellement depuis Gemini.
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Écriture
    # ------------------------------------------------------------------

    def upsert(self, chunk_ids: List[str], texts: List[str],
               embeddings: List[List[float]], metadatas: List[dict]) -> None:
        """Ajoute ou met à jour des chunks."""
        if not chunk_ids:
            return
        # ChromaDB ne supporte que des types primitifs dans les metadata
        cleaned = [_clean_metadata(m) for m in metadatas]
        self._collection.upsert(
            ids=chunk_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=cleaned,
        )

    def delete(self, chunk_ids: List[str]) -> None:
        if not chunk_ids:
            return
        self._collection.delete(ids=chunk_ids)

    def reset(self):
        """Vide complètement la collection."""
        try:
            self._client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Lecture
    # ------------------------------------------------------------------

    def search(self, query_embedding: List[float], k: int = 10) -> List[SearchHit]:
        """Recherche vectorielle top-k."""
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.count()),
        )
        hits: List[SearchHit] = []
        if not result["ids"] or not result["ids"][0]:
            return hits
        for chunk_id, doc, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            # Distance cosinus → similarité (1 - distance)
            score = 1.0 - float(dist)
            hits.append(SearchHit(
                chunk_id=chunk_id,
                text=doc,
                score=score,
                metadata=meta or {},
            ))
        return hits

    def get_all(self) -> list[SearchHit]:
        """Retourne tous les chunks (pour construire l'index BM25)."""
        if self.count() == 0:
            return []
        result = self._collection.get(include=["documents", "metadatas"])
        hits: List[SearchHit] = []
        for chunk_id, doc, meta in zip(
            result["ids"],
            result["documents"],
            result["metadatas"],
        ):
            hits.append(SearchHit(
                chunk_id=chunk_id,
                text=doc,
                score=0.0,
                metadata=meta or {},
            ))
        return hits

    def count(self) -> int:
        return self._collection.count()


def _clean_metadata(meta: dict) -> dict:
    """ChromaDB n'accepte que str/int/float/bool dans les metadata."""
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = ", ".join(str(x) for x in v)
        else:
            out[k] = str(v)
    return out

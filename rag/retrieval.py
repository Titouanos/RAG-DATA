"""Retrieval hybride : vector (Chroma) + lexical (BM25) fusionnés par RRF.

Pourquoi hybride : les embeddings sont forts sur la similarité sémantique mais
ratent les matches exacts (codes d'erreur, noms d'outils, acronymes). BM25
rattrape ça. La fusion RRF (Reciprocal Rank Fusion) donne un bon score combiné
sans avoir à tuner des poids.
"""

from __future__ import annotations

import logging
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List

from rag.store import ChromaStore, SearchHit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BM25 store
# ---------------------------------------------------------------------------


class BM25Store:
    """Index BM25 persisté sur disque via pickle.

    Reconstruit à chaque ingestion complète depuis les chunks du vector store.
    Peu coûteux même sur plusieurs milliers de chunks.
    """

    def __init__(self, path: Path):
        self.path = path
        self._bm25 = None
        self._chunk_ids: List[str] = []
        self._texts: List[str] = []

    def build(self, chunks: list[SearchHit]):
        """Construit l'index depuis tous les chunks."""
        from rank_bm25 import BM25Okapi

        self._chunk_ids = [c.chunk_id for c in chunks]
        self._texts = [c.text for c in chunks]

        if not self._texts:
            self._bm25 = None
            self.save()
            return

        tokenized = [_tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(tokenized)
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({
                "chunk_ids": self._chunk_ids,
                "texts": self._texts,
                "bm25": self._bm25,
            }, f)

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            with open(self.path, "rb") as f:
                data = pickle.load(f)
            self._chunk_ids = data["chunk_ids"]
            self._texts = data["texts"]
            self._bm25 = data["bm25"]
            return True
        except Exception as exc:
            logger.warning("Impossible de charger BM25 : %s", exc)
            return False

    def search(self, query: str, k: int = 20) -> List[SearchHit]:
        """Retourne les top-k chunks par score BM25."""
        if self._bm25 is None or not self._chunk_ids:
            return []
        tokenized = _tokenize(query)
        if not tokenized:
            return []

        scores = self._bm25.get_scores(tokenized)
        # Top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        hits: List[SearchHit] = []
        for i in top_indices:
            if scores[i] <= 0:
                break
            hits.append(SearchHit(
                chunk_id=self._chunk_ids[i],
                text=self._texts[i],
                score=float(scores[i]),
                metadata={},  # metadata récupérée depuis Chroma dans HybridRetriever
            ))
        return hits


# Tokenizer simple, multilingue, gère les mots composés et numériques
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+", re.UNICODE)
_STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
    "et", "ou", "mais", "donc", "or", "ni", "car", "que", "qui", "quoi",
    "où", "comment", "quand", "pourquoi", "est", "sont", "a", "ai", "as",
    "avoir", "être", "fait", "faire", "pour", "par", "avec", "sans",
    "dans", "sur", "sous", "ce", "cette", "ces", "cet", "mon", "ma", "mes",
    "ton", "ta", "tes", "son", "sa", "ses", "il", "elle", "on", "nous",
    "vous", "ils", "elles", "je", "tu", "me", "te", "se", "y", "en",
    "à", "aussi", "si", "non", "oui", "pas", "plus", "ne", "la", "là",
}
_STOPWORDS_EN = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "of", "in", "on", "at", "to", "for", "with", "from", "by", "about",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
    "they", "what", "which", "who", "whom", "whose", "when", "where", "why",
    "how", "not", "no", "so",
}
_STOPWORDS = _STOPWORDS_FR | _STOPWORDS_EN


def _tokenize(text: str) -> list[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Hybrid retriever : vector + BM25 + RRF
# ---------------------------------------------------------------------------


def rrf_fuse(*ranked_lists: list[str], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion.

    Chaque ranked_list est une liste d'IDs ordonnée par pertinence.
    Retourne [(id, score)] par score décroissant.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    fused_score: float
    vector_score: float | None = None
    bm25_score: float | None = None


class HybridRetriever:
    """Combine Chroma (vector) + BM25 avec fusion RRF."""

    def __init__(self, vector_store: ChromaStore, bm25_store: BM25Store,
                 embedder, rrf_k: int = 60):
        self.vector = vector_store
        self.bm25 = bm25_store
        self.embedder = embedder
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 20) -> List[RetrievedChunk]:
        """Recherche hybride top-k pour une seule query."""
        return self._search_single(query, k=k)

    def multi_search(self, queries: list[str], k: int = 20,
                     k_per_query: int | None = None) -> List[RetrievedChunk]:
        """Recherche multi-query : exécute le retrieval sur plusieurs queries
        dérivées et fusionne les résultats avec RRF.

        Utile pour les questions transversales (ex: "quels sont les types de X")
        qui demandent à recouvrir plusieurs sections de la doc, chacune
        traitant d'un type différent.

        Args:
            queries: liste des queries dérivées (la première est généralement
                     la query originale).
            k: nombre de résultats finaux à retourner (top-k global).
            k_per_query: nombre de résultats à récupérer par sous-query
                         (par défaut k pour avoir assez de matière à fusionner).
        """
        if not queries:
            return []
        if len(queries) == 1:
            return self._search_single(queries[0], k=k)

        if k_per_query is None:
            k_per_query = k

        # Pour chaque query, faire le retrieval hybride et collecter la liste
        # ordonnée des chunk_ids
        all_ranked_lists: list[list[str]] = []
        # On garde un cache global des SearchHit (vector + bm25) pour éviter
        # de re-fetch les chunks lors de la construction du résultat final.
        global_vec_map: dict[str, SearchHit] = {}
        global_bm25_map: dict[str, SearchHit] = {}

        for q in queries:
            # Vector
            try:
                q_emb = self.embedder.embed_query(q)
                vec_hits = self.vector.search(q_emb, k=k_per_query)
            except Exception as exc:
                logger.warning("Vector search failed for query %r : %s", q, exc)
                vec_hits = []
            # BM25
            try:
                bm25_hits = self.bm25.search(q, k=k_per_query)
            except Exception as exc:
                logger.warning("BM25 search failed for query %r : %s", q, exc)
                bm25_hits = []

            for h in vec_hits:
                global_vec_map.setdefault(h.chunk_id, h)
            for h in bm25_hits:
                global_bm25_map.setdefault(h.chunk_id, h)

            # Fusion intra-query (RRF entre vec et bm25 pour cette query)
            vec_ids = [h.chunk_id for h in vec_hits]
            bm25_ids = [h.chunk_id for h in bm25_hits]
            fused = rrf_fuse(vec_ids, bm25_ids, k=self.rrf_k)
            all_ranked_lists.append([cid for cid, _ in fused])

        # Fusion globale RRF entre les rankings de chaque query
        global_fused = rrf_fuse(*all_ranked_lists, k=self.rrf_k)

        # Pour les chunks qui ne sont dans aucune map, fetch depuis Chroma
        missing_ids = [
            cid for cid, _ in global_fused[:k]
            if cid not in global_vec_map and cid not in global_bm25_map
        ]
        if missing_ids:
            try:
                result = self.vector._collection.get(
                    ids=missing_ids,
                    include=["documents", "metadatas"],
                )
                for cid, doc, meta in zip(
                    result["ids"], result["documents"], result["metadatas"],
                ):
                    global_vec_map[cid] = SearchHit(
                        chunk_id=cid, text=doc, score=0.0, metadata=meta or {},
                    )
            except Exception as exc:
                logger.warning("Fetch manquants multi-query : %s", exc)

        # Construire les RetrievedChunk
        results: List[RetrievedChunk] = []
        for chunk_id, fused_score in global_fused[:k]:
            hit = global_vec_map.get(chunk_id) or global_bm25_map.get(chunk_id)
            if not hit:
                continue
            results.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=hit.text,
                metadata=hit.metadata or {},
                fused_score=fused_score,
                vector_score=global_vec_map[chunk_id].score if chunk_id in global_vec_map else None,
                bm25_score=global_bm25_map[chunk_id].score if chunk_id in global_bm25_map else None,
            ))
        return results

    def _search_single(self, query: str, k: int = 20) -> List[RetrievedChunk]:
        """Recherche hybride top-k."""
        # Vector search
        query_emb = self.embedder.embed_query(query)
        vec_hits = self.vector.search(query_emb, k=k)

        # BM25 search
        bm25_hits = self.bm25.search(query, k=k)

        # Fusion RRF sur les IDs
        vec_ids = [h.chunk_id for h in vec_hits]
        bm25_ids = [h.chunk_id for h in bm25_hits]
        fused = rrf_fuse(vec_ids, bm25_ids, k=self.rrf_k)

        # Construire le résultat avec metadata depuis vector store
        vec_map = {h.chunk_id: h for h in vec_hits}
        bm25_map = {h.chunk_id: h for h in bm25_hits}

        # Pour les chunks qui n'étaient que dans BM25, on va chercher metadata+texte
        missing_ids = [cid for cid, _ in fused if cid not in vec_map]
        if missing_ids:
            # Récupérer les chunks manquants depuis Chroma
            try:
                result = self.vector._collection.get(
                    ids=missing_ids,
                    include=["documents", "metadatas"],
                )
                for cid, doc, meta in zip(
                    result["ids"], result["documents"], result["metadatas"],
                ):
                    vec_map[cid] = SearchHit(
                        chunk_id=cid, text=doc, score=0.0, metadata=meta or {},
                    )
            except Exception as exc:
                logger.warning("Fetch manquants : %s", exc)

        results: List[RetrievedChunk] = []
        for chunk_id, fused_score in fused[:k]:
            hit = vec_map.get(chunk_id) or bm25_map.get(chunk_id)
            if not hit:
                continue
            results.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=hit.text,
                metadata=hit.metadata or {},
                fused_score=fused_score,
                vector_score=vec_map[chunk_id].score if chunk_id in vec_map else None,
                bm25_score=bm25_map[chunk_id].score if chunk_id in bm25_map else None,
            ))
        return results
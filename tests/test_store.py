"""Tests du QdrantStore hybride avec des vecteurs factices (pas de modèle chargé)."""

from __future__ import annotations

from rag_builder.core.embeddings.base import Embedding
from rag_builder.core.models import Chunk, EmbeddedChunk, SparseVector
from rag_builder.core.store import QdrantStore, point_id


def _mk(doc_id, order, dense, sidx, sval, text):
    return EmbeddedChunk(
        chunk=Chunk(text=text, order=order, page_or_section=f"p{order}"),
        dense=dense,
        sparse=SparseVector(indices=sidx, values=sval),
    )


def _seed(store: QdrantStore):
    store.ensure_collection("c", dense_dim=4, with_sparse=True)
    store.upsert_chunks(
        "c", doc_id="A", source_name="A.pdf", doc_type="pdf", content_hash="ha",
        embedded=[
            _mk("A", 0, [1, 0, 0, 0], [10, 20], [0.9, 0.5], "alpha teamcenter"),
            _mk("A", 1, [0, 1, 0, 0], [20, 30], [0.8, 0.4], "beta pdf"),
        ],
    )
    store.upsert_chunks(
        "c", doc_id="B", source_name="B.pdf", doc_type="pdf", content_hash="hb",
        embedded=[
            _mk("B", 0, [0, 0, 1, 0], [30, 40], [0.7, 0.3], "gamma"),
            _mk("B", 1, [0, 0, 0, 1], [40, 10], [0.6, 0.2], "delta"),
        ],
    )


def test_point_id_deterministic():
    assert point_id("A", 0) == point_id("A", 0)
    assert point_id("A", 0) != point_id("A", 1)
    assert point_id("A", 0) != point_id("B", 0)


def test_upsert_and_counts(store):
    _seed(store)
    assert store.count("c") == 4
    assert store.count_doc("c", "A") == 2
    assert store.list_doc_ids("c") == ["A", "B"]
    assert store.get_doc_hash("c", "A") == "ha"
    assert store.get_doc_hash("c", "absent") is None


def test_hybrid_search_ranks_expected_first(store):
    _seed(store)
    sparse = SparseVector(indices=[10, 20], values=[1.0, 0.5])
    q = Embedding(dense=[0.95, 0.05, 0, 0], sparse=sparse)
    hits = store.search("c", q, limit=3, with_sparse=True)
    assert hits, "au moins un hit attendu"
    assert hits[0].doc_id == "A"
    assert hits[0].payload["chunk_index"] == 0
    assert hits[0].text == "alpha teamcenter"


def test_dense_only_search(store):
    _seed(store)
    q = Embedding(dense=[0, 0, 1, 0], sparse=None)
    hits = store.search("c", q, limit=2, with_sparse=False)
    assert hits[0].doc_id == "B"


def test_delete_by_doc_id_removes_all_chunks(store):
    _seed(store)
    store.delete_by_doc_id("c", "A")
    assert store.count("c") == 2
    assert store.list_doc_ids("c") == ["B"]
    # Plus aucun chunk de A ne remonte, dense ou sparse.
    q = Embedding(dense=[1, 0, 0, 0], sparse=SparseVector(indices=[10, 20], values=[1.0, 0.5]))
    hits = store.search("c", q, limit=5, with_sparse=True)
    assert all(h.doc_id != "A" for h in hits)


def test_reingest_replaces_chunks(store):
    _seed(store)
    # Réingestion de A avec 1 seul chunk (doc rétréci) : delete puis upsert.
    store.delete_by_doc_id("c", "A")
    store.upsert_chunks(
        "c", doc_id="A", source_name="A.pdf", doc_type="pdf", content_hash="ha2",
        embedded=[_mk("A", 0, [1, 0, 0, 0], [10], [1.0], "alpha v2")],
    )
    assert store.count_doc("c", "A") == 1
    assert store.get_doc_hash("c", "A") == "ha2"

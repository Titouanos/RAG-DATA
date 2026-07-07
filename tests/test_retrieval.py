"""Test de la fusion RRF (utilitaire inter-requêtes)."""

from __future__ import annotations

from rag_builder.core.retrieval import rrf_fuse


def test_rrf_rewards_top_ranks_across_lists():
    list_a = ["x", "y", "z"]
    list_b = ["y", "x", "w"]
    fused = dict(rrf_fuse(list_a, list_b, k=60))
    # 'y' est 2e puis 1er, 'x' 1er puis 2e : scores proches, tous deux devant z/w.
    assert fused["x"] > fused["z"]
    assert fused["y"] > fused["w"]


def test_rrf_single_list_is_rank_order():
    fused = rrf_fuse(["a", "b", "c"], k=60)
    ids = [i for i, _ in fused]
    assert ids == ["a", "b", "c"]

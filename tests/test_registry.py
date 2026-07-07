"""Tests du registre de collections (JSON, embedding figé)."""

from __future__ import annotations

import pytest
from rag_builder.core.registry import CollectionError, CollectionRegistry


def test_create_get_list_delete(tmp_path):
    reg = CollectionRegistry(tmp_path / "collections.json")
    reg.create("docs", description="ma doc", top_k=7)
    assert reg.exists("docs")
    m = reg.require("docs")
    assert m.top_k == 7 and m.description == "ma doc"
    assert [c.name for c in reg.list()] == ["docs"]
    reg.delete("docs")
    assert not reg.exists("docs")


def test_duplicate_rejected(tmp_path):
    reg = CollectionRegistry(tmp_path / "c.json")
    reg.create("a")
    with pytest.raises(CollectionError):
        reg.create("a")


def test_invalid_name_rejected(tmp_path):
    reg = CollectionRegistry(tmp_path / "c.json")
    for bad in ["", "a b", "é", "a/b", "x" * 65]:
        with pytest.raises(CollectionError):
            reg.create(bad)


def test_embedding_model_is_frozen(tmp_path):
    reg = CollectionRegistry(tmp_path / "c.json")
    reg.create("a", embedding_model="BAAI/bge-m3", dense_dim=1024)
    # Modifier un réglage libre : OK.
    reg.update("a", top_k=9)
    assert reg.require("a").top_k == 9
    # Modifier l'embedding figé : refusé.
    with pytest.raises(CollectionError):
        reg.update("a", embedding_model="autre/modele")
    with pytest.raises(CollectionError):
        reg.update("a", dense_dim=768)


def test_persistence_across_instances(tmp_path):
    path = tmp_path / "c.json"
    CollectionRegistry(path).create("a", description="persist")
    reg2 = CollectionRegistry(path)
    assert reg2.require("a").description == "persist"

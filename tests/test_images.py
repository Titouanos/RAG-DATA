"""Tests de l'ImageStore (content-addressed, multi-collections, anti path-traversal)."""

from __future__ import annotations

import pytest
from rag_builder.core.images import INTERNAL_SCHEME, ImageStore


def test_save_is_content_addressed_and_idempotent(tmp_path):
    st = ImageStore(tmp_path)
    img = b"\x89PNG fake bytes"
    a = st.save("coll", "docX", img, ".png")
    b = st.save("coll", "docX", img, ".png")
    assert a.relative_path == b.relative_path  # même contenu → même chemin
    assert a.reference.startswith(f"{INTERNAL_SCHEME}coll/docX/")
    assert a.absolute_path.exists()


def test_resolve_roundtrip(tmp_path):
    st = ImageStore(tmp_path)
    stored = st.save("coll", "docX", b"data", ".png")
    resolved = st.resolve(stored.relative_path)
    assert resolved == stored.absolute_path


def test_resolve_blocks_traversal(tmp_path):
    st = ImageStore(tmp_path)
    assert st.resolve("../../etc/passwd") is None
    assert st.resolve("/etc/passwd") is None


def test_unsafe_ids_rejected(tmp_path):
    st = ImageStore(tmp_path)
    with pytest.raises(ValueError):
        st.save("../evil", "doc", b"x")
    with pytest.raises(ValueError):
        st.save("coll", "..", b"x")


def test_remove_doc_and_collection(tmp_path):
    st = ImageStore(tmp_path)
    st.save("coll", "d1", b"a", ".png")
    st.save("coll", "d1", b"b", ".png")
    assert st.remove_doc("coll", "d1") == 2
    st.save("coll", "d2", b"c", ".png")
    st.remove_collection("coll")
    assert not (tmp_path / "coll").exists()

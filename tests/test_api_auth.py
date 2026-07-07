"""Tests de l'API : authentification et contrôle d'accès (sans chargement de modèle)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from rag_builder.api import auth as auth_svc
from rag_builder.api.app import create_app
from rag_builder.config import Settings
from rag_builder.db.models import Role
from rag_builder.db.session import session_scope

ROOT = Path(__file__).resolve().parent.parent


def _settings(tmp_path) -> Settings:
    return Settings(
        storage_dir=tmp_path / "storage",
        data_dir=tmp_path / "data",
        models_cache_dir=ROOT / "storage" / "models_cache",
        warmup_on_start=False,  # pas de chargement de modèle pour les tests d'auth
    )


@pytest.fixture
def client(tmp_path):
    app = create_app(_settings(tmp_path))
    with TestClient(app) as c:
        # Bootstrap d'un admin et d'un user directement en base.
        engine = app.state.ctx.engine
        with session_scope(engine) as db:
            auth_svc.create_user(db, "admin", "admin-pass", role=Role.ADMIN)
            auth_svc.create_user(db, "bob", "bob-pass", role=Role.USER)
        yield c


def _login(client, username, password):
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r


def test_health_is_public(client):
    assert client.get("/health").status_code == 200


def test_unauthenticated_is_401(client):
    assert client.get("/collections").status_code == 401
    assert client.get("/auth/me").status_code == 401


def test_login_logout_me(client):
    _login(client, "admin", "admin-pass")
    me = client.get("/auth/me")
    assert me.status_code == 200 and me.json()["username"] == "admin"
    assert client.post("/auth/logout").status_code == 200
    assert client.get("/auth/me").status_code == 401


def test_bad_password_rejected(client):
    r = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
    assert r.status_code == 401


def test_admin_only_user_creation(client):
    # bob (user) ne peut pas créer de compte
    _login(client, "bob", "bob-pass")
    r = client.post("/auth/users", json={"username": "x", "password": "secret123"})
    assert r.status_code == 403
    client.post("/auth/logout")
    # admin peut
    _login(client, "admin", "admin-pass")
    r = client.post("/auth/users", json={"username": "carol", "password": "secret123"})
    assert r.status_code == 201 and r.json()["username"] == "carol"


def test_collection_creation_admin_only_by_default(client):
    _login(client, "bob", "bob-pass")
    r = client.post("/collections", json={"name": "kb"})
    assert r.status_code == 403  # collections_admin_only=True par défaut
    client.post("/auth/logout")
    _login(client, "admin", "admin-pass")
    r = client.post("/collections", json={"name": "kb", "rerank_enabled": False})
    assert r.status_code == 201, r.text
    assert r.json()["name"] == "kb"
    # doublon
    assert client.post("/collections", json={"name": "kb"}).status_code == 409

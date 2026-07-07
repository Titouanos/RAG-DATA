"""Authentification : mots de passe argon2 + sessions serveur (cookie httpOnly).

Sessions stockées en base (table `sessions`) → révocation possible (logout, expiration).
Le cookie ne porte qu'un token opaque aléatoire.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from sqlmodel import Session as DBSession
from sqlmodel import select

from rag_builder.db.models import Role, Session, User

SESSION_COOKIE = "ragb_session"
SESSION_TTL_DAYS = 7

_ph = PasswordHasher()


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _ph.verify(password_hash, password)
    except VerifyMismatchError:
        return False
    except Exception:  # noqa: BLE001 — hash corrompu / format inattendu
        return False


def create_user(db: DBSession, username: str, password: str, *, role: str = Role.USER) -> User:
    """Crée un utilisateur (lève ValueError si le nom est pris)."""
    username = username.strip()
    if not username or not password:
        raise ValueError("username et password requis")
    existing = db.exec(select(User).where(User.username == username)).first()
    if existing is not None:
        raise ValueError(f"L'utilisateur existe déjà : {username!r}")
    user = User(username=username, password_hash=hash_password(password), role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate(db: DBSession, username: str, password: str) -> User | None:
    user = db.exec(select(User).where(User.username == username)).first()
    if user is None or not user.is_active:
        return None
    if not verify_password(user.password_hash, password):
        return None
    return user


def create_session(db: DBSession, user_id: int, *, ttl_days: int = SESSION_TTL_DAYS) -> str:
    token = secrets.token_urlsafe(32)
    row = Session(
        token=token,
        user_id=user_id,
        expires_at=datetime.now(UTC) + timedelta(days=ttl_days),
    )
    db.add(row)
    db.commit()
    return token


def user_for_token(db: DBSession, token: str) -> User | None:
    """Retourne l'utilisateur d'un token de session valide (non expiré, actif)."""
    if not token:
        return None
    row = db.exec(select(Session).where(Session.token == token)).first()
    if row is None:
        return None
    expires = row.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    if expires < datetime.now(UTC):
        db.delete(row)
        db.commit()
        return None
    user = db.get(User, row.user_id)
    if user is None or not user.is_active:
        return None
    return user


def delete_session(db: DBSession, token: str) -> None:
    row = db.exec(select(Session).where(Session.token == token)).first()
    if row is not None:
        db.delete(row)
        db.commit()


def count_users(db: DBSession) -> int:
    return len(db.exec(select(User.id)).all())

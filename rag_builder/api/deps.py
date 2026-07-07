"""Dépendances FastAPI : contexte, session DB, utilisateur courant, contrôle de rôle."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, HTTPException, Request, status
from sqlmodel import Session

from rag_builder.api.auth import SESSION_COOKIE, user_for_token
from rag_builder.api.context import AppContext
from rag_builder.core.rag_service import RagService
from rag_builder.db.models import Role, User


def get_context(request: Request) -> AppContext:
    return request.app.state.ctx


def get_service(request: Request) -> RagService:
    return request.app.state.ctx.service


def get_db(request: Request) -> Iterator[Session]:
    engine = request.app.state.ctx.engine
    with Session(engine) as session:
        yield session


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get(SESSION_COOKIE, "")
    user = user_for_token(db, token)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Non authentifié")
    return user


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != Role.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Réservé aux admins")
    return user


def require_collection_manager(
    request: Request, user: User = Depends(current_user)
) -> User:
    """Création/suppression de collections : admin si `collections_admin_only` (défaut)."""
    settings = request.app.state.ctx.settings
    if settings.collections_admin_only and user.role != Role.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Création/suppression de collections réservée aux admins",
        )
    return user

"""Authentification : login/logout/me + gestion des comptes (admin)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlmodel import Session, select

from rag_builder.api import auth as auth_svc
from rag_builder.api.deps import current_user, get_db, require_admin
from rag_builder.api.schemas import CreateUserRequest, LoginRequest, UserOut
from rag_builder.db.models import User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=UserOut)
def login(
    body: LoginRequest,
    response: Response,
    request: Request,
    db: Session = Depends(get_db),
) -> UserOut:
    user = auth_svc.authenticate(db, body.username, body.password)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Identifiants invalides")
    settings = request.app.state.ctx.settings
    token = auth_svc.create_session(db, user.id, ttl_days=settings.session_ttl_days)
    response.set_cookie(
        auth_svc.SESSION_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        max_age=settings.session_ttl_days * 86400,
    )
    return UserOut(id=user.id, username=user.username, role=user.role)


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)) -> dict:
    token = request.cookies.get(auth_svc.SESSION_COOKIE, "")
    if token:
        auth_svc.delete_session(db, token)
    response.delete_cookie(auth_svc.SESSION_COOKIE)
    return {"status": "ok"}


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)) -> UserOut:
    return UserOut(id=user.id, username=user.username, role=user.role)


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    body: CreateUserRequest,
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> UserOut:
    try:
        user = auth_svc.create_user(db, body.username, body.password, role=body.role)
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return UserOut(id=user.id, username=user.username, role=user.role)


@router.get("/users", response_model=list[UserOut])
def list_users(_admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.exec(select(User).order_by(User.username)).all()
    return [UserOut(id=u.id, username=u.username, role=u.role) for u in users]

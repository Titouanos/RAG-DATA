"""Moteur SQLite (mode WAL) et gestion des sessions SQLModel.

WAL + `check_same_thread=False` : l'API (threads uvicorn) et le worker (thread dédié)
partagent la même base sans se bloquer. `init_db` crée les tables au démarrage (les
migrations Alembic sont une piste v2).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine


def make_engine(db_path: Path) -> Engine:
    """Crée le moteur SQLite en mode WAL."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _rec):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

    return engine


def init_db(engine: Engine) -> None:
    """Crée les tables si absentes. Importe les modèles pour peupler le metadata."""
    from rag_builder.db import models  # noqa: F401  (enregistre les tables)

    SQLModel.metadata.create_all(engine)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Contexte transactionnel : commit à la sortie, rollback en cas d'erreur."""
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

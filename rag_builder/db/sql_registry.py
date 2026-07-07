"""Registre de collections adossé à SQLite (remplace le JSON de la Phase 1).

Implémente la même interface que `core.registry.CollectionRegistry` et retourne des
`CollectionMeta`, de sorte que `RagService` fonctionne sans changement.
"""

from __future__ import annotations

from dataclasses import fields

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from rag_builder.core.registry import COLLECTION_NAME_RE, CollectionError, CollectionMeta
from rag_builder.db.models import Collection

_FROZEN = ("embedder", "embedding_model", "dense_dim", "supports_sparse")
_META_FIELDS = {f.name for f in fields(CollectionMeta)}


def _to_meta(row: Collection) -> CollectionMeta:
    return CollectionMeta(
        **{
            k: v
            for k, v in row.model_dump().items()
            if k in _META_FIELDS
        }
    )


def _validate_name(name: str) -> None:
    if not COLLECTION_NAME_RE.match(name):
        raise CollectionError(
            f"Nom de collection invalide : {name!r} (attendu : [A-Za-z0-9_-], 1-64 car.)"
        )


class SqlCollectionRegistry:
    """Persistance SQLite des métadonnées de collections."""

    def __init__(self, engine: Engine):
        self._engine = engine

    def exists(self, name: str) -> bool:
        with Session(self._engine) as s:
            return s.get(Collection, name) is not None

    def get(self, name: str) -> CollectionMeta | None:
        with Session(self._engine) as s:
            row = s.get(Collection, name)
            return _to_meta(row) if row else None

    def require(self, name: str) -> CollectionMeta:
        meta = self.get(name)
        if meta is None:
            raise CollectionError(f"Collection introuvable : {name!r}")
        return meta

    def list(self) -> list[CollectionMeta]:
        with Session(self._engine) as s:
            rows = s.exec(select(Collection).order_by(Collection.name)).all()
            return [_to_meta(r) for r in rows]

    def create(self, name: str, *, created_by: int | None = None, **overrides) -> CollectionMeta:
        _validate_name(name)
        # `created_by` n'est pas un champ de CollectionMeta ; on le stocke en base seulement.
        valid = {k: v for k, v in overrides.items() if k in _META_FIELDS and k != "name"}
        with Session(self._engine) as s:
            if s.get(Collection, name) is not None:
                raise CollectionError(f"La collection existe déjà : {name!r}")
            row = Collection(name=name, created_by=created_by, **valid)
            s.add(row)
            s.commit()
            s.refresh(row)
            return _to_meta(row)

    def update(self, name: str, **changes) -> CollectionMeta:
        with Session(self._engine) as s:
            row = s.get(Collection, name)
            if row is None:
                raise CollectionError(f"Collection introuvable : {name!r}")
            for frozen in _FROZEN:
                if frozen in changes and changes[frozen] != getattr(row, frozen):
                    raise CollectionError(
                        f"Le champ {frozen!r} est figé à la création de la collection "
                        f"(changer d'embedding = réindexation complète)."
                    )
            for key, value in changes.items():
                if key in _META_FIELDS and key != "name":
                    setattr(row, key, value)
            s.add(row)
            s.commit()
            s.refresh(row)
            return _to_meta(row)

    def delete(self, name: str) -> None:
        with Session(self._engine) as s:
            row = s.get(Collection, name)
            if row is not None:
                s.delete(row)
                s.commit()

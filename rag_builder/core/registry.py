"""Registre des collections RAG et de leurs métadonnées.

Chaque collection fige son **modèle d'embedding** à la création (en changer imposerait une
réindexation). Le registre porte aussi les réglages surchargeables par collection
(provider/modèle LLM, top_k, rerank, prompt système).

Phase 1 : persistance JSON (`storage/collections.json`). Phase 2 : remplacé par la table
SQLite `collections` — l'interface publique (`create/get/list/delete/update`) reste stable.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path

# Nom de collection sûr (= nom de collection Qdrant + segment de chemin d'images).
COLLECTION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class CollectionError(RuntimeError):
    """Erreur liée à une collection (déjà existante, introuvable, nom invalide)."""


@dataclass
class CollectionMeta:
    """Métadonnées et réglages d'une collection RAG."""

    name: str
    description: str = ""
    # Embedding figé à la création
    embedder: str = "local_bge_m3"
    embedding_model: str = "BAAI/bge-m3"
    dense_dim: int = 1024
    supports_sparse: bool = True
    # Réglages surchargeables
    rerank_enabled: bool = True  # reranker ONNX rapide par défaut (cf. rerank_model)
    rerank_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    top_k: int = 5
    rerank_k: int = 10  # 10 candidats → rerank CPU sous ~800 ms
    llm_provider: str = "mistral"
    llm_model: str = "mistral-large-latest"
    system_prompt: str | None = None
    ocr_enabled: bool = False
    created_at: str = ""

    @classmethod
    def _from_dict(cls, data: dict) -> CollectionMeta:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def _validate_name(name: str) -> None:
    if not COLLECTION_NAME_RE.match(name):
        raise CollectionError(
            f"Nom de collection invalide : {name!r} (attendu : [A-Za-z0-9_-], 1-64 car.)"
        )


class CollectionRegistry:
    """Persistance JSON des métadonnées de collections (thread-safe)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._items: dict[str, CollectionMeta] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for name, data in raw.items():
                self._items[name] = CollectionMeta._from_dict(data)
        except Exception as exc:  # noqa: BLE001
            raise CollectionError(
                f"Registre de collections illisible : {self.path} ({exc})"
            ) from exc

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: asdict(meta) for name, meta in self._items.items()}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)  # écriture atomique

    # ------------------------------------------------------------------

    def exists(self, name: str) -> bool:
        return name in self._items

    def get(self, name: str) -> CollectionMeta | None:
        return self._items.get(name)

    def require(self, name: str) -> CollectionMeta:
        meta = self._items.get(name)
        if meta is None:
            raise CollectionError(f"Collection introuvable : {name!r}")
        return meta

    def list(self) -> list[CollectionMeta]:
        return sorted(self._items.values(), key=lambda m: m.name)

    def create(self, name: str, **overrides) -> CollectionMeta:
        _validate_name(name)
        with self._lock:
            if name in self._items:
                raise CollectionError(f"La collection existe déjà : {name!r}")
            meta = CollectionMeta(
                name=name,
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                **overrides,
            )
            self._items[name] = meta
            self._save()
            return meta

    def update(self, name: str, **changes) -> CollectionMeta:
        with self._lock:
            meta = self.require(name)
            # Le modèle d'embedding est figé : on interdit sa modification ici.
            for frozen in ("embedder", "embedding_model", "dense_dim", "supports_sparse"):
                if frozen in changes and changes[frozen] != getattr(meta, frozen):
                    raise CollectionError(
                        f"Le champ {frozen!r} est figé à la création de la collection "
                        f"(changer d'embedding = réindexation complète)."
                    )
            for key, value in changes.items():
                if hasattr(meta, key):
                    setattr(meta, key, value)
            self._save()
            return meta

    def delete(self, name: str) -> None:
        with self._lock:
            if name in self._items:
                del self._items[name]
                self._save()

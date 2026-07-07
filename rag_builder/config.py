"""Configuration globale de RAG Builder (variables d'environnement + `.env`).

Ces réglages sont les **défauts au niveau application**. Les paramètres propres à une
collection (modèle d'embedding figé, provider LLM, top_k, rerank, prompt système) sont
portés par la collection elle-même (registre en Phase 1, table SQLite en Phase 2) et
surchargent ces défauts au moment de la requête.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du projet = parent du package rag_builder/
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Réglages chargés depuis l'environnement et `.env` (à la racine du projet)."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Chemins ---
    storage_dir: Path = Field(default=ROOT_DIR / "storage")
    data_dir: Path = Field(default=ROOT_DIR / "data")

    # --- Qdrant ---
    # local  : QdrantClient(path=...) embarqué (dev / poste)
    # server : QdrantClient(url=...) (déploiement Docker)
    qdrant_mode: str = Field(default="local")
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = Field(default=None)

    # --- Embeddings ---
    # local_bge_m3 (défaut, 100% local) | gemini | mistral
    embedder: str = Field(default="local_bge_m3")
    dense_model: str = Field(default="BAAI/bge-m3")
    dense_dim: int = Field(default=1024)
    # Cache des poids de modèles locaux (bge-m3, reranker). Pré-embarqué en prod.
    models_cache_dir: Path = Field(default=ROOT_DIR / "storage" / "models_cache")
    # En production réseau filtré : True → interdit tout téléchargement HF au runtime.
    hf_offline: bool = Field(default=False)

    # --- Reranking ---
    # Désactivé par défaut : bge-reranker-v2-m3 (568M) sur CPU met ~15 s pour 20 candidats,
    # très au-delà du budget de 800 ms (mesures Phase 1). Activable par collection.
    rerank_enabled: bool = Field(default=False)
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3")

    # --- Retrieval ---
    rerank_k: int = Field(default=20)  # candidats récupérés avant rerank
    top_k: int = Field(default=5)  # chunks finaux passés au LLM
    rrf_k: int = Field(default=60)  # constante de la fusion RRF

    # --- Chunking ---
    chunk_target_tokens: int = Field(default=500)
    chunk_overlap_tokens: int = Field(default=60)
    chunk_min_chars: int = Field(default=100)

    # --- Génération (LLM) ---
    # gemini (défaut Phase 1, réutilise la clé existante) | mistral | anthropic | ollama
    llm_provider: str = Field(default="gemini")
    llm_model: str = Field(default="gemini-2.5-flash")
    gemini_api_key: str | None = Field(default=None)
    mistral_api_key: str | None = Field(default=None)
    anthropic_api_key: str | None = Field(default=None)
    ollama_base_url: str = Field(default="http://localhost:11434")

    # --- Vision (description d'images, optionnel) ---
    vision_enabled: bool = Field(default=False)
    vision_model: str = Field(default="gemini-2.5-flash-lite")

    # --- Divers ---
    max_upload_mb: int = Field(default=100)
    debug_timing: bool = Field(default=False)

    @property
    def qdrant_path(self) -> Path:
        """Répertoire de la base Qdrant embarquée (mode local)."""
        return self.storage_dir / "qdrant"

    @property
    def images_dir(self) -> Path:
        """Racine du stockage des images extraites (par collection en dessous)."""
        return self.storage_dir / "images"

    @property
    def app_db_path(self) -> Path:
        """Fichier SQLite applicatif (utilisé à partir de la Phase 2)."""
        return self.storage_dir / "app.db"

    def ensure_dirs(self) -> None:
        """Crée les répertoires de travail s'ils n'existent pas."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_cache_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Retourne l'instance de configuration (mémoïsée)."""
    return Settings()

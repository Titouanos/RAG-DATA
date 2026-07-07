"""Chargement de la configuration depuis config.yaml et .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration centrale du RAG."""

    # Secrets
    gemini_api_key: str

    # Paths
    data_dir: Path
    storage_dir: Path
    root_dir: Path

    # Modèles
    embedding_model: str
    embedding_dim: int
    generation_model: str
    vision_model: str
    reranker_model: str

    # Chunking
    chunk_target_tokens: int
    chunk_overlap_tokens: int
    chunk_min_chars: int

    # Retrieval
    rerank_k: int
    top_k: int
    rrf_k: int
    use_reranker: bool

    # YouTube
    youtube_langs: list[str]
    youtube_group_seconds: int

    # MindMap
    mindmap_describe: bool
    mindmap_cache: bool

    # Video
    video_enable: bool
    video_model: str
    video_cache: bool
    video_timeout: int

    # Multimodal images : extraction et stockage des images des docs
    images_extract_pdf: bool
    images_extract_mindmap: bool
    images_extract_office: bool

    # Embedding
    embedding_batch_size: int

    # Rate limiting : overrides par modèle (dict: {model_name: {rpm, rpd}})
    rate_limits_overrides: dict[str, dict[str, int]] = field(default_factory=dict)

    # Paramètres internes utiles
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def chroma_dir(self) -> Path:
        return self.storage_dir / "chroma"

    @property
    def doc_state_path(self) -> Path:
        return self.storage_dir / "doc_state.json"

    @property
    def bm25_path(self) -> Path:
        return self.storage_dir / "bm25.pkl"

    @property
    def image_cache_dir(self) -> Path:
        return self.storage_dir / "image_cache"

    @property
    def video_cache_dir(self) -> Path:
        return self.storage_dir / "video_cache"


def load_config(config_path: str | Path = "config.yaml") -> Config:
    """Charge la config depuis config.yaml et les secrets depuis .env."""

    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY n'est pas défini dans .env. "
            "Copie .env.example vers .env et renseigne ta clé."
        )

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_dir = root / raw["paths"]["data_dir"]
    storage_dir = root / raw["paths"]["storage_dir"]

    # Créer les dossiers s'ils n'existent pas
    data_dir.mkdir(parents=True, exist_ok=True)
    storage_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        gemini_api_key=api_key,
        root_dir=root,
        data_dir=data_dir,
        storage_dir=storage_dir,
        embedding_model=raw["models"]["embedding"],
        embedding_dim=int(raw["models"]["embedding_dim"]),
        generation_model=raw["models"]["generation"],
        vision_model=raw["models"]["vision"],
        reranker_model=raw["models"]["reranker"],
        chunk_target_tokens=int(raw["chunking"]["target_tokens"]),
        chunk_overlap_tokens=int(raw["chunking"]["overlap_tokens"]),
        chunk_min_chars=int(raw["chunking"]["min_chunk_chars"]),
        rerank_k=int(raw["retrieval"]["rerank_k"]),
        top_k=int(raw["retrieval"]["top_k"]),
        rrf_k=int(raw["retrieval"]["rrf_k"]),
        use_reranker=bool(raw["retrieval"]["use_reranker"]),
        youtube_langs=list(raw["youtube"]["preferred_languages"]),
        youtube_group_seconds=int(raw["youtube"]["segment_group_seconds"]),
        mindmap_describe=bool(raw["mindmap"]["describe_images"]),
        mindmap_cache=bool(raw["mindmap"]["cache_descriptions"]),
        video_enable=bool(raw.get("video", {}).get("enable", True)),
        video_model=str(raw.get("video", {}).get("model") or raw["models"].get("video") or raw["models"]["generation"]),
        video_cache=bool(raw.get("video", {}).get("cache_transcriptions", True)),
        video_timeout=int(raw.get("video", {}).get("processing_timeout_seconds", 600)),
        images_extract_pdf=bool(raw.get("images", {}).get("extract_pdf", True)),
        images_extract_mindmap=bool(raw.get("images", {}).get("extract_mindmap", True)),
        images_extract_office=bool(raw.get("images", {}).get("extract_office", True)),
        embedding_batch_size=int(raw["embedding_batch"]["batch_size"]),
        rate_limits_overrides=raw.get("rate_limits") or {},
        raw=raw,
    )

    cfg.image_cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.video_cache_dir.mkdir(parents=True, exist_ok=True)
    return cfg

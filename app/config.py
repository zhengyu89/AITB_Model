from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.services.embedder import DEFAULT_MODEL_NAME


@dataclass(frozen=True)
class Settings:
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)
    )
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "malaysia_landmarks")
    )
    api_key: str | None = field(default_factory=lambda: os.getenv("API_KEY") or None)
    attraction_checkpoint: Path = field(
        default_factory=lambda: Path(os.getenv("ATTRACTION_CHECKPOINT", "my_landmark_attraction.pth"))
    )
    food_checkpoint: Path = field(
        default_factory=lambda: Path(os.getenv("FOOD_CHECKPOINT", "my_landmark_food.pth"))
    )
    default_device: str | None = field(default_factory=lambda: os.getenv("MODEL_DEVICE") or None)
    default_topk: int = field(default_factory=lambda: int(os.getenv("DEFAULT_TOPK", "5")))
    global_search_limit: int = field(default_factory=lambda: int(os.getenv("GLOBAL_SEARCH_LIMIT", "50")))
    accept_score: float = field(default_factory=lambda: float(os.getenv("ACCEPT_SCORE", "0.40")))
    tentative_score: float = field(default_factory=lambda: float(os.getenv("TENTATIVE_SCORE", "0.28")))
    min_gap: float = field(default_factory=lambda: float(os.getenv("MIN_GAP", "0.03")))


def get_settings() -> Settings:
    return Settings()


def get_embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)

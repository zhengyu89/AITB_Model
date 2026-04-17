from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.services.embedder import DEFAULT_MODEL_NAME


@dataclass(frozen=True)
class Settings:
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "malaysia_landmarks")
    attraction_checkpoint: Path = Path(os.getenv("ATTRACTION_CHECKPOINT", "my_landmark_attraction.pth"))
    food_checkpoint: Path = Path(os.getenv("FOOD_CHECKPOINT", "my_landmark_food.pth"))
    default_device: str | None = os.getenv("MODEL_DEVICE") or None
    default_topk: int = int(os.getenv("DEFAULT_TOPK", "5"))
    global_search_limit: int = int(os.getenv("GLOBAL_SEARCH_LIMIT", "50"))
    default_user_radius_m: float = float(os.getenv("DEFAULT_USER_RADIUS_M", "5000"))
    accept_score: float = float(os.getenv("ACCEPT_SCORE", "0.40"))
    tentative_score: float = float(os.getenv("TENTATIVE_SCORE", "0.28"))
    min_gap: float = float(os.getenv("MIN_GAP", "0.03"))


def get_settings() -> Settings:
    return Settings()


def get_embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image
from qdrant_client import QdrantClient

from app.config import ATTRACTION_CHECKPOINT, FOOD_CHECKPOINT, get_settings
from app.services.classifier import (
    load_head_branch,
    load_landmark_classifier,
    predict_from_embedding,
    query_embedding_from_pil,
    resolve_checkpoint_path,
)
from app.services.embedder import DinoV2Embedder
from app.services.geo_ranking import GeoPoint, accuracy_factor, age_factor, rank_and_decide
from app.services.qdrant_retrieval import aggregate_qdrant_results, qdrant_topk


@dataclass
class PredictionBundle:
    embedder: DinoV2Embedder
    device: str
    attraction: tuple | None
    food: tuple | None
    qdrant_client: QdrantClient
    qdrant_collection: str


def _existing_path(path: Path) -> Path | None:
    try:
        return resolve_checkpoint_path(path)
    except FileNotFoundError:
        return None


@lru_cache(maxsize=1)
def get_prediction_bundle() -> PredictionBundle:
    settings = get_settings()
    attraction_path = _existing_path(ATTRACTION_CHECKPOINT)
    food_path = _existing_path(FOOD_CHECKPOINT)
    if attraction_path is None and food_path is None:
        raise FileNotFoundError("At least one checkpoint is required: attraction or food.")

    base_path = attraction_path or food_path
    assert base_path is not None
    ckpt0, embedder, head0, device = load_landmark_classifier(base_path)

    attraction = None
    if attraction_path is not None:
        if attraction_path == base_path:
            attraction = (ckpt0, head0)
        else:
            attraction = load_head_branch(attraction_path, embedder, device)

    food = None
    if food_path is not None:
        if food_path == base_path:
            food = (ckpt0, head0)
        else:
            food = load_head_branch(food_path, embedder, device)

    return PredictionBundle(
        embedder=embedder,
        device=device,
        attraction=attraction,
        food=food,
        qdrant_client=QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key),
        qdrant_collection=settings.qdrant_collection,
    )


def predict_image(
    pil_image: Image.Image,
    topk: int,
    geo_point: GeoPoint | None = None,
    include_classification: bool = True,
    include_debug: bool = False,
) -> dict:
    bundle = get_prediction_bundle()
    settings = get_settings()
    embedding = query_embedding_from_pil(pil_image, bundle.embedder)
    vector = embedding.tolist()

    attraction_rows = None
    if include_classification and bundle.attraction is not None:
        ckpt, head = bundle.attraction
        attraction_rows = predict_from_embedding(embedding, ckpt, head, bundle.device, topk=topk)

    food_rows = None
    if include_classification and bundle.food is not None:
        ckpt, head = bundle.food
        food_rows = predict_from_embedding(embedding, ckpt, head, bundle.device, topk=topk)

    # Retrieval is always global; GPS only re-ranks the grouped places afterwards.
    global_rows = qdrant_topk(
        client=bundle.qdrant_client,
        collection=bundle.qdrant_collection,
        query_vector=vector,
        limit=max(topk, settings.global_search_limit),
    )
    grouped = aggregate_qdrant_results(global_rows)
    ranking = rank_and_decide(grouped, geo_point, settings)
    geo_applied = ranking.retrieval_scope == "geo_reranked"

    response = {
        "status": ranking.decision["status"],
        "retrieval_scope": ranking.retrieval_scope,
        "geo_reason": ranking.geo_reason,
        "geo_weight": ranking.geo_weight,
        "final_match": _build_final_match(ranking.decision, geo_applied),
        "candidates": _build_candidates(ranking.order[:topk], geo_applied),
        "classification": _build_classification_summary(attraction_rows, food_rows) if include_classification else None,
    }

    if include_debug:
        # Never echo the geo point itself; weights and per-candidate distances are enough.
        response["debug"] = {
            "embedding_model": bundle.embedder.model_name,
            "embedding_dim": bundle.embedder.embedding_dim,
            "device": bundle.device,
            "thresholds": {
                "accept_score": settings.accept_score,
                "tentative_score": settings.tentative_score,
                "min_gap": settings.min_gap,
            },
            "geo": {
                "provided": geo_point is not None,
                "legacy_fields": bool(geo_point and geo_point.legacy),
                "accuracy_factor": accuracy_factor(geo_point.accuracy_m) if geo_point else None,
                "age_factor": age_factor(geo_point.age_s) if geo_point else None,
                "weight": ranking.geo_weight,
                "max_weight": settings.geo_max_weight,
                "prior_distance_m": settings.geo_prior_distance_m,
                "reorder_window": settings.geo_reorder_window,
                "reorder_window_size": ranking.reorder_window_size,
                "visual_order": [item.get("display_name") for item in grouped[:topk]],
                "candidate_count": len(grouped),
            },
            "decision": _debug_decision(ranking.decision),
        }

    return response


def _debug_decision(decision: dict) -> dict:
    row = decision.get("row")
    return {
        "status": decision["status"],
        "name": row.get("display_name") if row else None,
        "score": decision["score"],
        "gap": decision["gap"],
        "hit_count": decision["hit_count"],
        "combined_score": row.get("combined_score") if row else None,
    }


def _build_final_match(decision: dict, geo_applied: bool) -> dict | None:
    row = decision.get("row")
    if row is None:
        return None
    payload = row.get("payload") or {}
    return {
        "name": row.get("display_name"),
        "category": row.get("category"),
        "class_path": row.get("class_path"),
        "similarity": float(decision["score"]),
        "reference_hits": int(decision["hit_count"]),
        "description": payload.get("description"),
        "location": payload.get("location"),
        "image_path": row.get("best_image_path"),
        "distance_m": row.get("distance_m") if geo_applied else None,
    }


def _build_candidates(rows: list[dict], geo_applied: bool) -> list[dict]:
    candidates: list[dict] = []
    for row in rows:
        payload = row.get("payload") or {}
        candidates.append(
            {
                "name": row.get("display_name"),
                "category": row.get("category"),
                "class_path": row.get("class_path"),
                "similarity": float(row.get("best_score") or 0.0),
                "combined_score": row.get("combined_score") if geo_applied else None,
                "reference_hits": int(row.get("hit_count") or 0),
                "description": payload.get("description"),
                "location": payload.get("location"),
                "distance_m": row.get("distance_m") if geo_applied else None,
            }
        )
    return candidates


def _build_classification_summary(attraction_rows: list[dict] | None, food_rows: list[dict] | None) -> dict:
    return {
        "attraction_top1": _top1_summary(attraction_rows),
        "food_top1": _top1_summary(food_rows),
    }


def _top1_summary(rows: list[dict] | None) -> dict | None:
    if not rows:
        return None
    top = rows[0]
    return {
        "name": top.get("display_name"),
        "class_path": top.get("class_path"),
        "probability": float(top.get("probability") or 0.0),
    }

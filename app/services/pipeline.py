from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image
from qdrant_client import QdrantClient

from app.config import Settings, get_settings
from app.services.classifier import (
    load_head_branch,
    load_landmark_classifier,
    predict_from_embedding,
    query_embedding_from_pil,
    resolve_checkpoint_path,
)
from app.services.embedder import DinoV2Embedder
from app.services.qdrant_retrieval import aggregate_qdrant_results, filter_rows_by_geo, qdrant_topk


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
    attraction_path = _existing_path(settings.attraction_checkpoint)
    food_path = _existing_path(settings.food_checkpoint)
    if attraction_path is None and food_path is None:
        raise FileNotFoundError("At least one checkpoint is required: attraction or food.")

    base_path = attraction_path or food_path
    assert base_path is not None
    ckpt0, embedder, head0, device = load_landmark_classifier(base_path, settings.default_device)

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
        qdrant_client=QdrantClient(url=settings.qdrant_url),
        qdrant_collection=settings.qdrant_collection,
    )


def predict_image(
    pil_image: Image.Image,
    topk: int,
    user_lat: float | None = None,
    user_lon: float | None = None,
    user_radius_m: float | None = None,
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

    global_rows = qdrant_topk(
        client=bundle.qdrant_client,
        collection=bundle.qdrant_collection,
        query_vector=vector,
        limit=max(topk, settings.global_search_limit),
    )
    global_grouped = aggregate_qdrant_results(global_rows)

    local_rows: list[dict] = []
    local_grouped: list[dict] = []
    used_scope = "global"
    radius_m = user_radius_m if user_radius_m is not None else settings.default_user_radius_m

    if user_lat is not None and user_lon is not None:
        local_rows = qdrant_topk(
            client=bundle.qdrant_client,
            collection=bundle.qdrant_collection,
            query_vector=vector,
            limit=max(topk, settings.global_search_limit),
            user_lat=user_lat,
            user_lon=user_lon,
            user_radius_m=radius_m,
        )
        if not local_rows:
            local_rows = filter_rows_by_geo(global_rows, user_lat, user_lon, radius_m)
        local_grouped = aggregate_qdrant_results(local_rows)
        if local_grouped and float(local_grouped[0]["best_score"]) >= settings.tentative_score:
            used_scope = "local"

    selected_rows = local_rows if used_scope == "local" else global_rows
    selected_grouped = local_grouped if used_scope == "local" else global_grouped
    final_decision = _decide_final(selected_grouped, settings)

    response = {
        "status": final_decision["status"],
        "retrieval_scope": used_scope,
        "final_match": _build_final_match(final_decision),
        "candidates": _build_candidates(selected_grouped[:topk]),
        "classification": _build_classification_summary(attraction_rows, food_rows) if include_classification else None,
    }

    if include_debug:
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
                "user_lat": user_lat,
                "user_lon": user_lon,
                "user_radius_m": radius_m,
                "local_candidate_count": len(local_grouped),
                "global_candidate_count": len(global_grouped),
            },
            "decision": final_decision,
        }

    return response


def _decide_final(grouped_rows: list[dict], settings: Settings) -> dict:
    if grouped_rows:
        top = grouped_rows[0]
        score = float(top["best_score"])
        second_score = float(grouped_rows[1]["best_score"]) if len(grouped_rows) > 1 else 0.0
        gap = score - second_score

        if score >= settings.accept_score and gap >= settings.min_gap:
            return {
                "status": "accept",
                "row": top,
                "score": score,
                "gap": gap,
                "hit_count": int(top["hit_count"]),
            }

        if score >= settings.tentative_score:
            return {
                "status": "tentative",
                "row": top,
                "score": score,
                "gap": gap,
                "hit_count": int(top["hit_count"]),
            }

    return {
        "status": "reject",
        "row": None,
        "score": None,
        "gap": None,
        "hit_count": 0,
    }


def _build_final_match(decision: dict) -> dict | None:
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
        "distance_m": row.get("best_distance_m"),
    }


def _build_candidates(rows: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    for row in rows:
        payload = row.get("payload") or {}
        candidates.append(
            {
                "name": row.get("display_name"),
                "category": row.get("category"),
                "class_path": row.get("class_path"),
                "similarity": float(row.get("best_score") or 0.0),
                "reference_hits": int(row.get("hit_count") or 0),
                "description": payload.get("description"),
                "location": payload.get("location"),
                "distance_m": row.get("best_distance_m"),
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

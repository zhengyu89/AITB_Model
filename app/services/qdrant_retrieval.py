from __future__ import annotations

from collections import defaultdict
import math

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, GeoPoint, GeoRadius, MatchValue


def qdrant_topk(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    limit: int,
    category: str | None = None,
    user_lat: float | None = None,
    user_lon: float | None = None,
    user_radius_m: float | None = None,
    geo_key: str = "location",
) -> list[dict]:
    conditions = []
    if category:
        conditions.append(FieldCondition(key="category", match=MatchValue(value=category)))
    if user_lat is not None and user_lon is not None and user_radius_m is not None:
        conditions.append(
            FieldCondition(
                key=geo_key,
                geo_radius=GeoRadius(center=GeoPoint(lat=user_lat, lon=user_lon), radius=user_radius_m),
            )
        )

    query_filter = Filter(must=conditions) if conditions else None

    response = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False,
    )
    rows: list[dict] = []
    for point in response.points:
        payload = point.payload or {}
        rows.append(
            {
                "score": float(point.score) if point.score is not None else 0.0,
                "display_name": payload.get("display_name"),
                "class_name": payload.get("class_name"),
                "class_path": payload.get("class_path"),
                "image_path": payload.get("image_path"),
                "category": payload.get("category"),
                "payload": payload,
            }
        )
    return rows


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def filter_rows_by_geo(
    rows: list[dict],
    user_lat: float,
    user_lon: float,
    user_radius_m: float,
) -> list[dict]:
    filtered: list[dict] = []
    for row in rows:
        payload = row.get("payload") or {}
        location = payload.get("location") or {}
        lat = location.get("lat", payload.get("lat")) if isinstance(location, dict) else payload.get("lat")
        lon = location.get("lon", payload.get("lon")) if isinstance(location, dict) else payload.get("lon")
        coverage = payload.get("coverage_radius_m")
        if lat is None or lon is None:
            continue

        try:
            distance_m = haversine_distance_m(float(user_lat), float(user_lon), float(lat), float(lon))
        except (TypeError, ValueError):
            continue

        accepted = distance_m <= float(user_radius_m)
        if not accepted and coverage is not None:
            try:
                accepted = distance_m <= float(coverage)
            except (TypeError, ValueError):
                accepted = False

        if accepted:
            row_copy = row.copy()
            row_copy["distance_m"] = distance_m
            filtered.append(row_copy)

    filtered.sort(
        key=lambda row: (
            row.get("distance_m") if row.get("distance_m") is not None else 1e18,
            -float(row.get("score") or 0.0),
        )
    )
    return filtered


def aggregate_qdrant_results(rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = defaultdict(
        lambda: {
            "display_name": "",
            "class_name": "",
            "class_path": "",
            "category": None,
            "best_score": 0.0,
            "total_score": 0.0,
            "hit_count": 0,
            "best_image_path": None,
            "best_distance_m": None,
            "payload": {},
        }
    )

    for row in rows:
        key = str(row.get("class_path") or row.get("display_name") or "")
        score = float(row.get("score") or 0.0)
        item = grouped[key]
        item["display_name"] = row.get("display_name") or item["display_name"]
        item["class_name"] = row.get("class_name") or item["class_name"]
        item["class_path"] = row.get("class_path") or item["class_path"]
        item["category"] = row.get("category") or item["category"]
        item["total_score"] += score
        item["hit_count"] += 1
        if score >= float(item["best_score"]):
            item["best_score"] = score
            item["best_image_path"] = row.get("image_path")
            item["best_distance_m"] = row.get("distance_m")
            item["payload"] = row.get("payload") or item["payload"]

    items = list(grouped.values())
    items.sort(
        key=lambda item: (float(item["best_score"]), float(item["total_score"]), int(item["hit_count"])),
        reverse=True,
    )
    for item in items:
        payload = item.get("payload") or {}
        item["description"] = payload.get("description")
        item["location"] = payload.get("location")
    return items

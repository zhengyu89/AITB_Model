from __future__ import annotations

import argparse
import math
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from dinov2_embedder import DEFAULT_MODEL_NAME, DinoV2Embedder


QDRANT_URL = "http://localhost:6333"
COLLECTION = "malaysia_landmarks_dinov2"
TOPK = 100
DEFAULT_USER_RADIUS_M = 5000


client = QdrantClient(url=QDRANT_URL)
embedder = None


def get_embedder(device: Optional[str] = None):
    global embedder
    if embedder is None:
        embedder = DinoV2Embedder(model_name=DEFAULT_MODEL_NAME, device=device)
    return embedder


def embed_image(path: str, device: Optional[str] = None):
    return get_embedder(device=device).embed_path(path)


def haversine_distance_m(lat1, lon1, lat2, lon2):
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def query_similar_images(
    query_image_path: str,
    city: Optional[str] = None,
    limit: int = TOPK,
    device: Optional[str] = None,
):
    q_vec = embed_image(query_image_path, device=device)

    filter_expr = None
    if city:
        filter_expr = Filter(must=[FieldCondition(key="city", match=MatchValue(value=city))])

    response = client.query_points(
        collection_name=COLLECTION,
        query=q_vec.tolist(),
        limit=limit,
        query_filter=filter_expr,
        with_payload=True,
        with_vectors=False,
    )

    return [
        {
            "id": point.id,
            "score": point.score,
            "payload": point.payload or {},
        }
        for point in response.points
    ]


def query_with_geo_filter(
    query_image_path: str,
    user_lat: Optional[float],
    user_lon: Optional[float],
    user_radius_m: Optional[float] = None,
    city: Optional[str] = None,
):
    if user_radius_m is None:
        user_radius_m = DEFAULT_USER_RADIUS_M

    candidates = []
    for h in query_similar_images(query_image_path, city=city, limit=TOPK):
        payload = h["payload"]
        lat = payload.get("lat")
        lon = payload.get("lon")
        coverage = payload.get("coverage_radius_m")
        if lat is None or lon is None:
            continue

        dist = None
        if user_lat is not None and user_lon is not None:
            dist = haversine_distance_m(user_lat, user_lon, lat, lon)

        accept = False
        if dist is not None:
            if dist <= user_radius_m:
                accept = True
            elif coverage is not None and dist <= coverage:
                accept = True

        if accept:
            candidates.append(
                {
                    "id": h["id"],
                    "score": h["score"],
                    "payload": payload,
                    "distance_m": dist,
                }
            )

    candidates = sorted(
        candidates,
        key=lambda item: (
            item["distance_m"] if item["distance_m"] is not None else 1e9,
            -item["score"],
        ),
    )
    return candidates


def parse_args():
    parser = argparse.ArgumentParser(description="Query the DINOv2 Qdrant landmark index.")
    parser.add_argument("--image", required=True, help="Path to the query image.")
    parser.add_argument("--city", default=None, help="Optional city filter.")
    parser.add_argument("--user-lat", type=float, default=None, help="User latitude for geo filtering.")
    parser.add_argument("--user-lon", type=float, default=None, help="User longitude for geo filtering.")
    parser.add_argument(
        "--user-radius-m",
        type=float,
        default=DEFAULT_USER_RADIUS_M,
        help="Accepted user radius in meters when geo filtering is enabled.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of nearest results to print.")
    parser.add_argument("--device", default=None, help="Torch device override, for example 'cuda'.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.user_lat is not None and args.user_lon is not None:
        results = query_with_geo_filter(
            args.image,
            args.user_lat,
            args.user_lon,
            user_radius_m=args.user_radius_m,
            city=args.city,
        )
        print("Filtered candidates (within radius):")
        for result in results[: args.limit]:
            print(
                result["payload"].get("display_name"),
                "dist_m=",
                result["distance_m"],
                "score=",
                result["score"],
                "path=",
                result["payload"].get("image_path"),
            )
    else:
        results = query_similar_images(
            args.image,
            city=args.city,
            limit=args.limit,
            device=args.device,
        )
        print("Nearest candidates:")
        for result in results:
            print(
                result["payload"].get("display_name"),
                "score=",
                result["score"],
                "path=",
                result["payload"].get("image_path"),
            )

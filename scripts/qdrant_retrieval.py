"""Qdrant vector search for reference images (same collection as ingest_images_to_qdrant)."""
from __future__ import annotations

from qdrant_client import QdrantClient


def qdrant_topk(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    limit: int,
) -> list[dict]:
    response = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    rows: list[dict] = []
    for point in response.points:
        pl = point.payload or {}
        rows.append(
            {
                "score": float(point.score) if point.score is not None else 0.0,
                "display_name": pl.get("display_name"),
                "class_path": pl.get("class_path"),
                "image_path": pl.get("image_path"),
                "category": pl.get("category"),
            }
        )
    return rows

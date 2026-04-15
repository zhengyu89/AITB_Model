from __future__ import annotations

import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from dinov2_embedder import DEFAULT_MODEL_NAME, SUPPORTED_EXTENSIONS, DinoV2Embedder


COLLECTION = "malaysia_landmarks_dinov2"
DATA_DIR = Path("data/reference")
BATCH_UPSERT = 32
QDRANT_URL = "http://localhost:6333"
DEFAULT_COVERAGE_M = 500
MODEL_NAME = DEFAULT_MODEL_NAME


def collect_points(data_dir: Path):
    points = []
    for class_dir in sorted(path for path in data_dir.rglob("*") if path.is_dir()):
        image_files = [
            child
            for child in sorted(class_dir.iterdir())
            if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not image_files:
            continue

        class_name = class_dir.name
        class_path = str(class_dir.relative_to(data_dir))
        category = class_dir.parent.name if class_dir.parent != data_dir else None
        display_name = class_name.replace("_", " ").title()
        class_meta = {}
        meta_file = class_dir / "metadata.json"
        if meta_file.exists():
            class_meta = json.loads(meta_file.read_text())

        for img_path in image_files:
            payload = {
                "class_name": class_name,
                "class_path": class_path,
                "category": category,
                "display_name": display_name,
                "city": class_meta.get("city"),
                "lat": class_meta.get("lat"),
                "lon": class_meta.get("lon"),
                "coverage_radius_m": class_meta.get("coverage_radius_m", DEFAULT_COVERAGE_M),
            }

            img_json = img_path.with_suffix(".json")
            if img_json.exists():
                try:
                    payload.update(json.loads(img_json.read_text()))
                except Exception:
                    pass

            points.append({"path": str(img_path), "payload": payload})

    return points


def main() -> int:
    client = QdrantClient(url=QDRANT_URL)
    embedder = DinoV2Embedder(model_name=MODEL_NAME)

    try:
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=embedder.embedding_dim, distance=Distance.COSINE),
        )
    except Exception as exc:
        print("recreate_collection:", exc)

    points = collect_points(DATA_DIR)
    batch = []

    for start in tqdm(range(0, len(points), BATCH_UPSERT), desc="Embedding & batching"):
        chunk = points[start : start + BATCH_UPSERT]
        vectors = embedder.embed_paths([item["path"] for item in chunk])

        for item, vector in zip(chunk, vectors):
            payload = item["payload"].copy()
            payload["image_path"] = item["path"]
            batch.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        client.upsert(collection_name=COLLECTION, points=batch)
        batch = []

    print("Ingest finished. Total points:", len(points))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

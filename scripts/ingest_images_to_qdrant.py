from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.config import get_embedding_model_name
from app.services.embedder import SUPPORTED_EXTENSIONS, DinoV2Embedder


COLLECTION = "malaysia_landmarks"
DATA_DIR = Path("data/reference")
ATTRACTIONS_CSV = Path("attractions200226.csv")
BATCH_UPSERT = 32
QDRANT_URL = "http://localhost:6333"
DEFAULT_COVERAGE_M = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed reference images and ingest them into Qdrant.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--collection", type=str, default=COLLECTION)
    parser.add_argument("--qdrant-url", type=str, default=QDRANT_URL)
    parser.add_argument("--batch-size", type=int, default=BATCH_UPSERT)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--attractions-csv", type=Path, default=ATTRACTIONS_CSV)
    return parser.parse_args()


def _normalize_name(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[’'`]", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _csv_aliases(name: str) -> set[str]:
    aliases = {_normalize_name(name)}
    for inner in re.findall(r"\(([^)]+)\)", name):
        aliases.add(_normalize_name(inner))
    outer = re.sub(r"\([^)]*\)", "", name).strip()
    if outer:
        aliases.add(_normalize_name(outer))
    return {alias for alias in aliases if alias}


def _to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_location_fields(payload: dict) -> None:
    location = payload.get("location")
    lat: float | None = None
    lon: float | None = None

    if isinstance(location, dict):
        lat = _to_float(location.get("lat"))
        lon = _to_float(location.get("lon"))

    if lat is None or lon is None:
        lat = _to_float(payload.get("lat"))
        lon = _to_float(payload.get("lon"))

    if lat is not None and lon is not None:
        payload["location"] = {"lat": lat, "lon": lon}
    else:
        payload.pop("location", None)

    payload.pop("lat", None)
    payload.pop("lon", None)


def load_attraction_metadata(csv_path: Path) -> dict[str, dict]:
    if not csv_path.exists():
        return {}

    rows_by_alias: dict[str, dict] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("name") or "").strip()
            if not name:
                continue
            lat = _to_float(row.get("latitude"))
            lon = _to_float(row.get("longitude"))
            payload = {
                "display_name": name,
                "description": (row.get("description") or "").strip() or None,
                "lat": lat,
                "lon": lon,
            }
            if lat is not None and lon is not None:
                payload["location"] = {"lat": lat, "lon": lon}
            for alias in _csv_aliases(name):
                rows_by_alias[alias] = payload
    return rows_by_alias


def collect_points(data_dir: Path, attraction_csv: Path):
    attraction_meta = load_attraction_metadata(attraction_csv)
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
        csv_meta = attraction_meta.get(_normalize_name(class_name), {}) if category == "attraction" else {}
        display_name = csv_meta.get("display_name") or class_name.replace("_", " ").title()
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
                "description": csv_meta.get("description"),
                "location": csv_meta.get("location"),
                "coverage_radius_m": class_meta.get("coverage_radius_m", DEFAULT_COVERAGE_M),
            }
            payload.update(class_meta)
            _normalize_location_fields(payload)

            img_json = img_path.with_suffix(".json")
            if img_json.exists():
                try:
                    payload.update(json.loads(img_json.read_text()))
                except Exception:
                    pass

            _normalize_location_fields(payload)

            points.append({"path": str(img_path), "payload": payload})

    return points


def main() -> int:
    args = parse_args()
    embedding_model_name = get_embedding_model_name()
    client = QdrantClient(url=args.qdrant_url)
    embedder = DinoV2Embedder(model_name=embedding_model_name, device=args.device)

    try:
        client.recreate_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=embedder.embedding_dim, distance=Distance.COSINE),
        )
    except Exception as exc:
        print("recreate_collection:", exc)

    points = collect_points(args.data_dir, args.attractions_csv)
    batch = []

    for start in tqdm(range(0, len(points), args.batch_size), desc="Embedding & batching"):
        chunk = points[start : start + args.batch_size]
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

        client.upsert(collection_name=args.collection, points=batch)
        batch = []

    print(
        "Ingest finished. Total points:",
        len(points),
        "collection:",
        args.collection,
        "model:",
        embedding_model_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Malaysia Landmark Recognition

A small Malaysia landmark and food recognition project built on top of **Meta DINOv2** (default: `facebook/dinov2-large`), **Qdrant** vector search, two **linear probe** classifiers, and a minimal **FastAPI** service. It also serves general-purpose sentence embeddings (`all-MiniLM-L6-v2` by default) over `POST /api/v1/embeddings`, so callers such as AI-Travel-Buddy's RAG search don't need to load their own copy of the model.

## Contents

- [Overview](#overview)
- [Key Points](#key-points)
- [Deployment Requirements](#deployment-requirements)
- [Data Layout](#data-layout)
- [Local Installation](#local-installation)
- [Project Structure](#project-structure)
- [Operation Guides](#operation-guides)
- [GPS-Aware Recognition](#gps-aware-recognition)
- [Text Embedding Service](#text-embedding-service)
- [Typical Workflow](#typical-workflow)
- [Main Outputs](#main-outputs)
- [Notes](#notes)

## Overview

```mermaid
flowchart TB
  subgraph data [Data]
    REF[data/reference leaf folders = classes]
  end
  subgraph app [Service Layer]
    DINO[app/services/embedder.py]
    CLS[app/services/classifier.py]
    RET[app/services/qdrant_retrieval.py]
    PIPE[app/services/pipeline.py]
  end
  subgraph online [Online Retrieval]
    ING[scripts/ingest_images_to_qdrant.py]
    QD[(Qdrant)]
    ING --> QD
  end
  subgraph train [Training]
    TL[scripts/train.py]
    PTH[(.pth checkpoints)]
    TL --> PTH
  end
  subgraph serve [Serving]
    API[app/main.py FastAPI]
    WEB[temp/webui.py Streamlit]
  end
  REF --> ING
  REF --> TL
  DINO --> ING
  DINO --> TL
  DINO --> API
  DINO --> WEB
  CLS --> API
  CLS --> WEB
  RET --> API
  RET --> WEB
  PIPE --> API
  QD --> API
  QD --> WEB
  PTH --> API
  PTH --> WEB
```

## Key Points

- The current training approach is **linear probe / transfer learning**.
- DINOv2 remains the shared feature extractor.
- Qdrant stores DINOv2 embeddings for reference images and serves retrieval results.
- The trained `.pth` files contain **classifier head weights**, not a standalone embedding backbone.
- Retrieval still uses DINOv2 embeddings. The linear probe checkpoints provide classification outputs for the attraction and food branches.

## Deployment Requirements

The following specifications are minimum practical baselines for serving predictions. They assume:

- one API container with one Uvicorn worker
- low-to-moderate request volume
- Qdrant is hosted separately
- the server performs inference only, not model training or bulk Qdrant ingestion

| Environment | CPU | RAM | GPU | Free SSD |
|---|---:|---:|---|---:|
| Linux UAT | 4 vCPU minimum, x86_64 | 8 GB | Optional for native CPU execution. If using the supplied Docker Compose configuration unchanged: CUDA-capable NVIDIA GPU with ≥6 GB VRAM | 20 GB |
| Linux production | 8 vCPU minimum, x86_64 | 16 GB | CUDA-capable NVIDIA GPU with ≥8 GB VRAM; 12–16 GB preferred | 50 GB |
| macOS UAT | Apple Silicon, 4+ CPU cores | 16 GB unified memory | Not required; current implementation runs inference on CPU | 20 GB |
| macOS production | Apple Silicon, 8+ CPU cores | 16 GB minimum; 24 GB preferred | Not required; current implementation runs inference on CPU | 50 GB |

Linux with an NVIDIA GPU is the recommended production platform. The supplied `docker-compose.yml` requests an NVIDIA GPU, so it will not run unchanged through Docker Desktop on macOS. The current device selection supports CUDA or CPU; it does not yet select Apple's Metal/MPS backend.

If Qdrant is installed on the same server, add at least 2 CPU cores, 4 GB RAM, and storage appropriate for the vector collection. Training or bulk ingestion should preferably run on a separate machine with at least 32 GB RAM and 12-16 GB GPU VRAM.

### Production Scaling

Scale the API horizontally behind a load balancer. Run one worker in each container and assign one GPU to each container:

```text
Load Balancer
    │
    ├── Container 1 → 1 worker → GPU 1
    ├── Container 2 → 1 worker → GPU 2
    └── Container 3 → 1 worker → GPU 3
```

This prevents multiple worker processes from competing for the same GPU and keeps model memory isolated per container.

The API runs with one Uvicorn worker per container. Scale by adding containers behind the load balancer rather than adding workers inside a container.

## Data Layout

```text
data/reference/
  attraction/<landmark_class_name>/*.jpg|png|webp|...
  food/<food_class_name>/*.jpg|png|webp|...
```

- Each leaf directory that directly contains images is treated as one class.
- Optional class-level metadata can be stored in `metadata.json`.
- Optional per-image metadata can be stored in a sibling `.json` file and will be merged into the Qdrant payload during ingestion.
- Attraction metadata can also be loaded from `attractions.csv`. The CSV is used to enrich payload fields such as `display_name`, `description`, and `location`.
- The embedding step still uses only images. CSV and JSON files are metadata sources only; they do not affect the image vector itself.

## Local Installation

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
```

Run all commands from the repository root.

## Project Structure

| Path | Purpose |
|------|---------|
| `app/main.py` | FastAPI entrypoint. |
| `app/services/embedder.py` | DINOv2 embedding service. |
| `app/services/classifier.py` | Checkpoint loading and classifier prediction helpers. |
| `app/services/qdrant_retrieval.py` | Qdrant retrieval and aggregation helpers. |
| `app/services/geo_ranking.py` | GPS re-ranking of grouped candidates (soft prior, guardrails, decision). |
| `app/services/pipeline.py` | Shared prediction pipeline for the API. |
| `app/services/text_embedder.py` | Loads the sentence embedding model once and encodes text for `/embeddings`. |
| `app/routes/predict.py` | `/predict` and `/predict/upload` request handling. |
| `app/routes/embeddings.py` | `/embeddings` request handling. |
| `scripts/train.py` | Trains a linear classifier head for attraction or food. |
| `scripts/ingest_images_to_qdrant.py` | Embeds reference images and writes them into Qdrant. |
| `scripts/pick_eval_images.py` | Copies a small evaluation sample set from `data/reference`. |
| `temp/webui.py` | Streamlit UI for manual testing. |
| `tests/` | Unit tests for geo re-ranking and the predict routes. |

## Operation Guides

### Train Linear Probe Heads

```bash
python scripts/train.py --subset-prefix attraction

python scripts/train.py --subset-prefix food
```

### Ingest Reference Images into Qdrant

```bash
python scripts/ingest_images_to_qdrant.py
```

If needed, point ingestion at a different attraction metadata file:

```bash
python scripts/ingest_images_to_qdrant.py \
  --attractions-csv attractions.csv
```

The embedding backbone is now read from `EMBEDDING_MODEL_NAME` and defaults to `facebook/dinov2-large`.

If your Qdrant deployment requires authentication, set `QDRANT_API_KEY` in the environment. The API service, ingestion script, and Streamlit UI all read the same optional key.

Make sure the same DINO backbone is used everywhere:

- training checkpoints
- Qdrant ingestion
- API inference
- Streamlit inference

### Run FastAPI

```bash
uvicorn app.main:app --reload
```

Or use the short launcher from the repo root:

```bash
python main.py
```

Public routes:

- `GET /health`

Protected API routes:

- `GET /api/v1/`
- `POST /api/v1/predict/`
- `POST /api/v1/predict/upload`
- `POST /api/v1/embeddings`
- `GET /api/v1/docs`
- `GET /api/v1/openapi.json`

### Container Deployment

Build the image from the repository root:

Method 1: pass environment variables directly with `docker run` and `docker build`:

```bash
docker build -f docker/Dockerfile -t malaysia-landmark-recognition .
```

```bash
docker run --rm -p 8000:8000 \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  -e QDRANT_API_KEY=your-qdrant-key \
  -e QDRANT_COLLECTION=malaysia_landmarks \
  -e VISION_SERVICE_API_KEY=your-secret-key \
  malaysia-landmark-recognition
```

Method 2: use Docker Compose:

```bash
cp .env.example .env
docker compose up -d --build
```

The provided `docker-compose.yml` reads variables from `.env`, requests all visible NVIDIA GPUs, and starts the same API container with one Uvicorn worker.

GPU notes:

- This requires Docker with NVIDIA GPU support on the host.
- The container prints a short CUDA self-check during startup so you can confirm whether `torch` can see the GPU.
- If your Qdrant deployment requires authentication, set `QDRANT_API_KEY`. Leave it empty for local unauthenticated setups.
- Set `VISION_SERVICE_API_KEY` to protect the `/api/v1` routes. Callers must send the same value in the `X-API-KEY` header. The main FastAPI backend (`AI-Travel-Buddy`) should reference this same value in its vision client configuration. If it is empty, `/api/v1` is public.
- `TEXT_EMBEDDING_MODEL_NAME` (default `all-MiniLM-L6-v2`) is downloaded from Hugging Face on first use unless it's already in the container's Hugging Face cache, same as the DINOv2 backbone.

Startup behavior:

- The container runs `docker/docker-entrypoint.sh`.
- It performs startup smoke checks and prints a CUDA status summary.
- It does not run Qdrant ingestion on startup.
- `docker compose up` is only responsible for starting the API container.

If you want to ingest reference images manually:

```bash
docker compose run --rm api python scripts/ingest_images_to_qdrant.py
```

If you want to force a full rebuild:

```bash
docker compose run --rm api python scripts/ingest_images_to_qdrant.py --rebuild
```


### Run Streamlit

```bash
./venv/bin/streamlit run temp/webui.py
```

### Build a Small Eval Set

```bash
python scripts/pick_eval_images.py --per-class 2 --seed 42
```

## GPS-Aware Recognition

GPS is a soft prior, never a filter. Retrieval against Qdrant is always global
(no radius filter, no distance-first sort); a device fix, when trusted, can
only re-rank grouped candidates that are already visually close to the top
match, and it can never turn a raw-sub-threshold candidate into a match.

### Request fields

Sent as JSON fields on `POST /api/v1/predict/`, or multipart form fields on
`POST /api/v1/predict/upload` (never as query parameters, so they don't end up
in the access log):

| Field | Required together | Meaning |
|---|---|---|
| `geo_lat`, `geo_lon` | yes | The device's current position (WGS84 decimal degrees). Only send this when there is evidence it also describes the photo's subject, e.g. an in-app camera capture with a fresh, accurate fix. Rejected with `422` if only one is set, out of range, non-finite, or `(0, 0)`. |
| `geo_accuracy_m` | no | Horizontal accuracy of the fix, in metres. Lower is trusted more. |
| `geo_age_s` | no | Age of the fix in seconds. Lower is trusted more. |
| `user_lat`, `user_lon` | yes | Deprecated legacy aliases for `geo_lat`/`geo_lon`. Still accepted (with unknown accuracy/age, so a low weight) for backward compatibility. On `/predict/upload` these remain query parameters, so avoid them for new integrations. |

### Response fields

| Field | Meaning |
|---|---|
| `retrieval_scope` | `global` when no geo point was applied, `geo_reranked` when it was. |
| `geo_reason` | Why geo did or didn't change the result: `not_provided`, `zero_weight`, `strong_visual_match`, `geo_disambiguated`, `reordered`, `no_effect`. |
| `geo_weight` | The effective geo prior weight used (`0` when not applied). |
| `final_match.distance_m`, `candidates[].distance_m` | Distance from the geo point. Present only when `retrieval_scope == "geo_reranked"`. |
| `candidates[].similarity` | Always the raw visual similarity, regardless of geo. |
| `candidates[].combined_score` | Similarity plus the weighted geo prior. Present only when geo was applied. |

### Algorithm

1. Qdrant returns the top `max(topk, GLOBAL_SEARCH_LIMIT)` hits by embedding similarity (no geo filter), grouped by place and sorted by visual similarity.
2. If no geo point was supplied, or its computed weight is `0`, the decision uses the visual order as-is (`retrieval_scope = "global"`).
3. Otherwise, only the places within `GEO_REORDER_WINDOW` similarity of the visual #1 (and still at or above `TENTATIVE_SCORE`) are eligible to be reordered, by `combined_score = similarity + geo_weight * exp(-distance_m / GEO_PRIOR_DISTANCE_M)`. Every other place keeps its visual-order position below them.
4. `geo_weight = GEO_MAX_WEIGHT * accuracy_factor(geo_accuracy_m) * age_factor(geo_age_s)`, where each factor is `1.0` at or below its "full" breakpoint (50 m / 60 s), `0.5` up to its "half" breakpoint (200 m / 120 s), `0` beyond it, and `0.5` when the value is missing (legacy calls).
5. Accept/tentative thresholds (`ACCEPT_SCORE`, `TENTATIVE_SCORE`) are always checked against **raw** similarity, never the combined score, so geo can disambiguate between two already-plausible places but can never promote a weak visual match.
6. The gap used for `MIN_GAP` is always between the #1 and #2 of whichever order (visual or combined) produced the decision — never a "nearest vs. next-nearest" comparison.

### Tuning

| Variable | Default | Meaning |
|---|---:|---|
| `GEO_MAX_WEIGHT` | `0.08` | Upper bound on how much the geo prior can add to a similarity score. |
| `GEO_PRIOR_DISTANCE_M` | `8000` | Distance (metres) at which the prior decays to `~0.37`. |
| `GEO_REORDER_WINDOW` | `0.04` | Similarity margin below the visual #1 that remains eligible for reordering. |

These, along with `ACCEPT_SCORE`, `TENTATIVE_SCORE`, `MIN_GAP`, and
`GLOBAL_SEARCH_LIMIT`, should be tuned against a labelled evaluation set (camera
photos with a known capture location, plus gallery photos taken elsewhere)
before relying on geo re-ranking in production.

Run the unit tests covering this logic with:

```bash
python -m unittest discover -s tests -t .
```

## Text Embedding Service

`POST /api/v1/embeddings` serves general-purpose sentence embeddings so that
other services (currently AI-Travel-Buddy's RAG search) don't each need to
load their own copy of the model. It requires the same `X-API-KEY` header as
the predict routes.

### Request

```json
{
  "texts": ["nasi lemak", "cendol"],
  "model": "all-MiniLM-L6-v2"
}
```

| Field | Required | Meaning |
|---|---|---|
| `texts` | yes | 1 to `TEXT_EMBEDDING_MAX_BATCH` (default 128) strings, embedded in order. Never logged. |
| `model` | no | Defaults to `TEXT_EMBEDDING_MODEL_NAME`. If set, must match it exactly or the request is rejected — this is a safety check, not a way to pick a different model at request time. |

### Response

```json
{
  "model": "all-MiniLM-L6-v2",
  "dimension": 384,
  "embeddings": [[...], [...]]
}
```

One vector per input text, in the same order. `dimension` is the model's
actual output length, not a configured value, so it can never drift from what
`embeddings` actually contains.

### Errors

| Status | When |
|---|---|
| `401` | Missing or wrong `X-API-KEY`. |
| `400` | `model` doesn't match `TEXT_EMBEDDING_MODEL_NAME`. |
| `422` | `texts` is empty or has more items than `TEXT_EMBEDDING_MAX_BATCH`. |
| `503` | The model failed to load. Retried automatically on the next request. |

### Behavior notes

- The model is loaded once at process startup (in the FastAPI `lifespan`) and
  reused for every request; encoding runs in a thread pool so it doesn't block
  the event loop. If the startup load fails (e.g. a network hiccup while
  downloading the model), the service logs a warning and keeps serving image
  recognition; `/embeddings` then retries the load on its next call.
- `encode()` is always called with the library's default arguments — no
  prefixes, no normalization, no other preprocessing — because the caller's
  Qdrant collection already stores vectors produced the same way. Changing
  `TEXT_EMBEDDING_MODEL_NAME` changes the vector space, so any existing
  collection must be re-embedded to match.
- Bake the model into the Docker image (or otherwise warm the Hugging Face
  cache) rather than relying on a first-request download in production.

## Typical Workflow

1. Put reference images under `data/reference/attraction/...` and `data/reference/food/...`.
2. Train the two linear probe heads.
3. Rebuild Qdrant manually using the same DINO backbone when reference data changes.
4. Start FastAPI or Streamlit.
5. Test predictions against real user images.

## Main Outputs

| Output | Description |
|--------|-------------|
| `my_landmark_attraction.pth` | Attraction classifier head checkpoint. |
| `my_landmark_food.pth` | Food classifier head checkpoint. |
| Qdrant collection | Reference image vectors plus payload metadata. |

## Notes

- The attraction and food classifier scores should not be compared directly across models.
- A linear probe checkpoint is **not** a replacement backbone.
- If you ever want the trained model itself to produce retrieval embeddings, that would require a different training strategy such as backbone fine-tuning or metric learning.
- New ingested points store a dedicated `location` object, read by GPS re-ranking (see [GPS-Aware Recognition](#gps-aware-recognition)). Radius-based filtering is no longer used.

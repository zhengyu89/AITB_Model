from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from PIL import Image

from app.config import get_settings
from app.services.pipeline import get_prediction_bundle, predict_image


app = FastAPI(title="Malaysia Landmark Recognition API", version="0.2.0")


class PredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image bytes.")
    topk: int = Field(default_factory=lambda: get_settings().default_topk, ge=1, le=20)
    user_lat: float | None = Field(default=None, description="Optional user latitude.")
    user_lon: float | None = Field(default=None, description="Optional user longitude.")
    user_radius_m: float | None = Field(default=None, gt=0, description="Optional local search radius in meters.")
    include_classification: bool = Field(default=True, description="Include attraction/food classifier summaries.")
    include_debug: bool = Field(default=False, description="Include debug-only fields.")


class GeoLocation(BaseModel):
    lat: float = Field(..., description="Latitude stored in the Qdrant geo payload.")
    lon: float = Field(..., description="Longitude stored in the Qdrant geo payload.")


class MatchResult(BaseModel):
    name: str | None = Field(default=None, description="Resolved display name of the matched class.")
    category: str | None = Field(default=None, description="Top-level bucket such as attraction or food.")
    class_path: str | None = Field(default=None, description="Internal class path used by the reference dataset.")
    similarity: float = Field(..., description="Best grouped Qdrant similarity score.")
    reference_hits: int = Field(..., description="Number of reference images grouped into this final match.")
    description: str | None = Field(default=None, description="Human-readable description loaded from Qdrant payload.")
    location: GeoLocation | None = Field(
        default=None,
        description="Geo payload loaded from Qdrant. Present when the matched point has geographic metadata.",
    )
    image_path: str | None = Field(default=None, description="Best matching reference image path.")
    distance_m: float | None = Field(
        default=None,
        description="Distance in meters from the user coordinates when geo filtering is applied.",
    )


class CandidateResult(BaseModel):
    name: str | None = Field(default=None, description="Candidate display name.")
    category: str | None = Field(default=None, description="Candidate bucket such as attraction or food.")
    class_path: str | None = Field(default=None, description="Internal class path for this candidate.")
    similarity: float = Field(..., description="Grouped similarity score for this candidate.")
    reference_hits: int = Field(..., description="Number of grouped reference images for this candidate.")
    description: str | None = Field(default=None, description="Description loaded from Qdrant payload.")
    location: GeoLocation | None = Field(
        default=None,
        description="Geo payload for this candidate when available in Qdrant.",
    )
    distance_m: float | None = Field(
        default=None,
        description="Distance in meters from the user coordinates when geo filtering is applied.",
    )


class ClassificationTop1(BaseModel):
    name: str | None = Field(default=None, description="Top-1 class name from the classifier branch.")
    class_path: str | None = Field(default=None, description="Top-1 classifier class path.")
    probability: float = Field(..., description="Softmax probability from the classifier branch.")


class ClassificationSummary(BaseModel):
    attraction_top1: ClassificationTop1 | None = Field(
        default=None,
        description="Top-1 summary from the attraction classifier branch.",
    )
    food_top1: ClassificationTop1 | None = Field(
        default=None,
        description="Top-1 summary from the food classifier branch.",
    )


class PredictResponse(BaseModel):
    status: Literal["accept", "tentative", "reject"] = Field(
        ...,
        description="Final decision state after retrieval scoring.",
    )
    retrieval_scope: Literal["local", "global"] = Field(
        ...,
        description="Whether the final decision came from GPS-local retrieval or global fallback.",
    )
    final_match: MatchResult | None = Field(
        default=None,
        description="Final grouped match selected by the pipeline, or null when the request is rejected.",
    )
    candidates: list[CandidateResult] = Field(
        default_factory=list,
        description="Top grouped retrieval candidates returned to the caller.",
    )
    classification: ClassificationSummary | None = Field(
        default=None,
        description="Optional classifier summaries from the attraction and food linear probe heads.",
    )
    debug: dict[str, Any] | None = Field(
        default=None,
        description="Optional debug payload included only when include_debug=true.",
    )


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "qdrant_url": settings.qdrant_url,
        "qdrant_collection": settings.qdrant_collection,
    }


@app.get("/config")
def config() -> dict:
    bundle = get_prediction_bundle()
    settings = get_settings()
    return {
        "embedding_model": bundle.embedder.model_name,
        "embedding_dim": bundle.embedder.embedding_dim,
        "device": bundle.device,
        "attraction_checkpoint": str(settings.attraction_checkpoint),
        "food_checkpoint": str(settings.food_checkpoint),
        "qdrant_collection": settings.qdrant_collection,
        "global_search_limit": settings.global_search_limit,
        "default_user_radius_m": settings.default_user_radius_m,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    image = _decode_base64_image(request.image_base64)
    return _run_prediction(
        image=image,
        topk=request.topk,
        user_lat=request.user_lat,
        user_lon=request.user_lon,
        user_radius_m=request.user_radius_m,
        include_classification=request.include_classification,
        include_debug=request.include_debug,
    )


@app.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(
    file: UploadFile = File(...),
    topk: int = Query(default=get_settings().default_topk, ge=1, le=20),
    user_lat: float | None = Query(default=None),
    user_lon: float | None = Query(default=None),
    user_radius_m: float | None = Query(default=None, gt=0),
    include_classification: bool = Query(default=True),
    include_debug: bool = Query(default=False),
) -> PredictResponse:
    try:
        content = await file.read()
        image = Image.open(BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    return _run_prediction(
        image=image,
        topk=topk,
        user_lat=user_lat,
        user_lon=user_lon,
        user_radius_m=user_radius_m,
        include_classification=include_classification,
        include_debug=include_debug,
    )


def _decode_base64_image(image_base64: str) -> Image.Image:
    try:
        payload = image_base64.split(",", 1)[-1]
        image_bytes = base64.b64decode(payload)
        return Image.open(BytesIO(image_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64 payload: {exc}") from exc


def _run_prediction(
    image: Image.Image,
    topk: int,
    user_lat: float | None,
    user_lon: float | None,
    user_radius_m: float | None,
    include_classification: bool,
    include_debug: bool,
) -> PredictResponse:
    try:
        return PredictResponse.model_validate(
            predict_image(
                pil_image=image,
                topk=topk,
                user_lat=user_lat,
                user_lon=user_lon,
                user_radius_m=user_radius_m,
                include_classification=include_classification,
                include_debug=include_debug,
            )
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

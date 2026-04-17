from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from PIL import Image

from app.config import get_settings
from app.services.pipeline import predict_image
from app.schema import PredictRequest, PredictResponse
from app.utils import decode_base64_image

app = FastAPI(title="Malaysia Landmark Recognition API", version="0.2.0")


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "qdrant_url": settings.qdrant_url,
        "qdrant_collection": settings.qdrant_collection,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    image = decode_base64_image(request.image_base64)
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

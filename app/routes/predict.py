from fastapi import APIRouter, File, Form, Query, UploadFile, HTTPException
from app.schema import PredictRequest, PredictResponse
from app.utils import decode_base64_image
from app.services.geo_ranking import GeoPoint, GeoPointError, build_geo_point
from app.services.pipeline import predict_image
from app.config import get_settings
from PIL import Image
from io import BytesIO


router = APIRouter(prefix="/predict", tags=["Predict"])

@router.post("/", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    image = decode_base64_image(request.image_base64)
    return _run_prediction(
        image=image,
        topk=request.topk,
        geo_point=_parse_geo_point(
            geo_lat=request.geo_lat,
            geo_lon=request.geo_lon,
            geo_accuracy_m=request.geo_accuracy_m,
            geo_age_s=request.geo_age_s,
            legacy_lat=request.user_lat,
            legacy_lon=request.user_lon,
        ),
        include_classification=request.include_classification,
        include_debug=request.include_debug,
    )


@router.post("/upload", response_model=PredictResponse)
async def predict_upload(
    file: UploadFile = File(...),
    geo_lat: float | None = Form(default=None),
    geo_lon: float | None = Form(default=None),
    geo_accuracy_m: float | None = Form(default=None),
    geo_age_s: float | None = Form(default=None),
    topk: int = Query(default=get_settings().default_topk, ge=1, le=20),
    # Legacy: query-string coordinates end up in access logs. Use the geo_* form fields instead.
    user_lat: float | None = Query(default=None, deprecated=True),
    user_lon: float | None = Query(default=None, deprecated=True),
    include_classification: bool = Query(default=True),
    include_debug: bool = Query(default=False),
) -> PredictResponse:
    geo_point = _parse_geo_point(
        geo_lat=geo_lat,
        geo_lon=geo_lon,
        geo_accuracy_m=geo_accuracy_m,
        geo_age_s=geo_age_s,
        legacy_lat=user_lat,
        legacy_lon=user_lon,
    )
    try:
        content = await file.read()
        image = Image.open(BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    return _run_prediction(
        image=image,
        topk=topk,
        geo_point=geo_point,
        include_classification=include_classification,
        include_debug=include_debug,
    )


def _parse_geo_point(**fields: float | None) -> GeoPoint | None:
    try:
        return build_geo_point(**fields)
    except GeoPointError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _run_prediction(
    image: Image.Image,
    topk: int,
    geo_point: GeoPoint | None,
    include_classification: bool,
    include_debug: bool,
) -> PredictResponse:
    try:
        return PredictResponse.model_validate(
            predict_image(
                pil_image=image,
                topk=topk,
                geo_point=geo_point,
                include_classification=include_classification,
                include_debug=include_debug,
            )
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

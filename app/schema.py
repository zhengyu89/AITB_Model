from pydantic import BaseModel, Field
from typing import Any, Literal

from app.config import get_settings


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
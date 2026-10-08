from __future__ import annotations

from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from app.services.embedder import pil_to_rgb
from app.services.pipeline import get_prediction_bundle

router = APIRouter(prefix="/embed", tags=["Embed"])


class EmbedResponse(BaseModel):
    embedding: list[float] = Field(..., description="L2-normalized DINOv2 embedding vector.")
    dim: int = Field(..., description="Embedding vector dimensionality.")
    model: str = Field(..., description="Name of the embedding model used.")


@router.post("/upload", response_model=EmbedResponse)
async def embed_upload(file: UploadFile = File(...)) -> EmbedResponse:
    """
    Return just the image embedding vector — no Qdrant write, no collection
    changes. Used by AI-Travel-Buddy's knowledge-base image sync, which
    assembles the full payload (area, place, description — already known
    from the Laravel admin panel) itself and performs the upsert.
    """
    try:
        content = await file.read()
        image = pil_to_rgb(Image.open(BytesIO(content)))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    bundle = get_prediction_bundle()
    vector = bundle.embedder.embed_pil_images([image])[0]

    return EmbedResponse(
        embedding=vector.tolist(),
        dim=len(vector),
        model=bundle.embedder.model_name,
    )

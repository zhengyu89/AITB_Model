from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.schema import EmbeddingsRequest, EmbeddingsResponse
from app.services.text_embedder import TextEncoderUnavailable, encode_texts

router = APIRouter(tags=["Embeddings"])


@router.post("/embeddings", response_model=EmbeddingsResponse)
async def create_embeddings(request: EmbeddingsRequest) -> EmbeddingsResponse:
    settings = get_settings()
    if request.model != settings.text_embedding_model_name:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")

    try:
        # CPU-bound: keep it off the event loop.
        vectors = await run_in_threadpool(encode_texts, request.texts)
    except TextEncoderUnavailable as exc:
        raise HTTPException(status_code=503, detail="Text embedding model is not ready.") from exc

    return EmbeddingsResponse(model=request.model, dimension=len(vectors[0]), embeddings=vectors)

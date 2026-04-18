from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routes.api import api_app
from app.services.pipeline import get_prediction_bundle


logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Preloading vision models and clients...")
    get_prediction_bundle()
    logger.info("Vision service preload complete.")
    yield


app = FastAPI(title="Malaysia Landmark Recognition", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.mount("/api/v1", api_app)

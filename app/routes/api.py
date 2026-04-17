from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routes.predict import router as predict_router

api_app = FastAPI(title="Malaysia Landmark Recognition API", version="1.0.0")


@api_app.middleware("http")
async def verify_api_key(request: Request, call_next):
    settings = get_settings()
    if settings.api_key is not None:
        provided_key = request.headers.get("X-API-KEY")
        if provided_key != settings.api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key."},
            )

    return await call_next(request)


@api_app.get("/")
def api_root() -> dict[str, str]:
    return {"message": "Welcome to the Malaysia Landmark Recognition API"}


api_app.include_router(predict_router)

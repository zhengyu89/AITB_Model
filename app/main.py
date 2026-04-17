from __future__ import annotations

from fastapi import FastAPI

from app.routes.api import api_app

app = FastAPI(title="Malaysia Landmark Recognition", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.mount("/api/v1", api_app)

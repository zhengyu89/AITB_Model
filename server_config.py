from __future__ import annotations

import os


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


APP_IMPORT = os.getenv("UVICORN_APP", "app.main:app")
HOST = os.getenv("UVICORN_HOST", "0.0.0.0")
PORT = int(os.getenv("UVICORN_PORT", "8000"))
RELOAD = _env_flag("UVICORN_RELOAD", True)
LOG_LEVEL = os.getenv("UVICORN_LOG_LEVEL", "info")

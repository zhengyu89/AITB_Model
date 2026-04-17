from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image
from fastapi import HTTPException


def decode_base64_image(image_base64: str) -> Image.Image:
    try:
        payload = image_base64.split(",", 1)[-1]
        image_bytes = base64.b64decode(payload)
        return Image.open(BytesIO(image_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64 payload: {exc}") from exc

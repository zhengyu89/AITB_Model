from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

from app.config import get_settings

class DinoV2Embedder:
    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        model_name = model_name or settings.embedding_model_name

        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise SystemExit(
                "transformers is not installed. Run `./venv/bin/pip install -r requirements.txt` first."
            ) from exc

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.processor = _load_with_local_fallback(AutoImageProcessor, model_name)
        self.model = _load_with_local_fallback(AutoModel, model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = int(self.model.config.hidden_size)

    def embed_path(self, image_path: str | Path) -> np.ndarray:
        return self.embed_paths([image_path])[0]

    def embed_paths(self, image_paths: Sequence[str | Path]) -> np.ndarray:
        images = [load_image_rgb(path) for path in image_paths]
        return self.embed_pil_images(images)

    def embed_pil_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)

        return embeddings.cpu().numpy().astype("float32")


def load_image_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


def pil_to_rgb(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _load_with_local_fallback(loader_cls, model_name: str):
    try:
        return loader_cls.from_pretrained(model_name)
    except OSError:
        return loader_cls.from_pretrained(model_name, local_files_only=True)

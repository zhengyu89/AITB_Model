from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from app.services.embedder import DinoV2Embedder, pil_to_rgb


_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class LoadedClassifier:
    checkpoint: dict
    head: nn.Module


def query_embedding_from_pil(pil_image: Image.Image, embedder: DinoV2Embedder) -> np.ndarray:
    with torch.no_grad():
        emb = embedder.embed_pil_images([pil_to_rgb(pil_image)])
    return emb[0].astype(np.float32)


def resolve_checkpoint_path(checkpoint: Path | str) -> Path:
    path = Path(checkpoint)
    if path.is_file():
        return path
    alt = _REPO_ROOT / path
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")


def load_landmark_classifier(checkpoint: Path | str, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = resolve_checkpoint_path(checkpoint)
    ckpt = torch.load(path, map_location=device)
    embedder = DinoV2Embedder(model_name=ckpt["dinov2_model_name"])
    head = nn.Linear(ckpt["embedding_dim"], ckpt["num_classes"]).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    return ckpt, embedder, head, device


def load_head_branch(checkpoint: Path | str, embedder: DinoV2Embedder, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = resolve_checkpoint_path(checkpoint)
    ckpt = torch.load(path, map_location=device)
    if ckpt["dinov2_model_name"] != embedder.model_name:
        raise ValueError(
            f"Checkpoint DINO {ckpt['dinov2_model_name']!r} != embedder {embedder.model_name!r}"
        )
    if int(ckpt["embedding_dim"]) != int(embedder.embedding_dim):
        raise ValueError("Checkpoint embedding_dim does not match embedder.embedding_dim")
    head = nn.Linear(ckpt["embedding_dim"], ckpt["num_classes"]).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    return ckpt, head


def predict_from_embedding(embedding: np.ndarray, ckpt, head, device, topk: int = 5) -> list[dict]:
    with torch.no_grad():
        query = torch.from_numpy(embedding).to(device).unsqueeze(0)
        logits = head(query).squeeze(0)
    return _topk_from_logits(logits, ckpt["class_paths"], topk)


def predict_pil_image(pil_image: Image.Image, ckpt, embedder, head, device, topk: int = 5) -> list[dict]:
    emb = query_embedding_from_pil(pil_image, embedder)
    return predict_from_embedding(emb, ckpt, head, device, topk=topk)


def _topk_from_logits(logits_1d: torch.Tensor, class_paths: list[str], topk: int) -> list[dict]:
    probs = torch.softmax(logits_1d, dim=0)
    k = min(topk, probs.numel())
    vals, idxs = torch.topk(probs, k)
    rows: list[dict] = []
    for prob, idx in zip(vals.tolist(), idxs.tolist()):
        class_path = class_paths[idx]
        display_name = class_path.split("/")[-1].replace("_", " ").title()
        rows.append(
            {
                "class_path": class_path,
                "display_name": display_name,
                "probability": prob,
            }
        )
    return rows

"""
Load my_landmark.pth and run predictions (shared by Web UI and CLI).

Training stays in train_landmark_head.py; this module is inference-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps

from dinov2_embedder import DinoV2Embedder


def query_embedding_from_pil(pil_image: Image.Image, embedder: DinoV2Embedder) -> np.ndarray:
    """Same preprocessing as predict_pil_image; returns (dim,) float32 (FAISS / Qdrant)."""
    pil_image = ImageOps.exif_transpose(pil_image)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    with torch.no_grad():
        emb = embedder.embed_pil_images([pil_image])
    return emb[0].astype(np.float32)


def load_landmark_classifier(checkpoint: Path | str, device: str | None = None):
    """Load frozen DINOv2 + trained linear head from my_landmark.pth (or custom path)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(checkpoint)
    if not path.is_file():
        alt = _REPO_ROOT / path
        if alt.is_file():
            path = alt
    ckpt = torch.load(path, map_location=device)
    embedder = DinoV2Embedder(model_name=ckpt["dinov2_model_name"], device=device)
    head = nn.Linear(ckpt["embedding_dim"], ckpt["num_classes"]).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    return ckpt, embedder, head, device


def load_head_branch(checkpoint: Path | str, embedder: DinoV2Embedder, device: str | None = None):
    """Load only the linear head; DINO must match embedder (same model + dim)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(checkpoint)
    if not path.is_file():
        alt = _REPO_ROOT / path
        if alt.is_file():
            path = alt
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


def _topk_from_logits(logits_1d: torch.Tensor, class_paths: list[str], topk: int) -> list[dict]:
    probs = torch.softmax(logits_1d, dim=0)
    k = min(topk, probs.numel())
    vals, idxs = torch.topk(probs, k)
    out: list[dict] = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        cp = class_paths[i]
        leaf = cp.split("/")[-1]
        display = leaf.replace("_", " ").title()
        out.append({"class_path": cp, "display_name": display, "probability": v})
    return out


def predict_from_embedding(embedding: np.ndarray, ckpt, head, device, topk: int = 5) -> list[dict]:
    """Linear head only; embedding is (dim,) float32 from query_embedding_from_pil."""
    with torch.no_grad():
        q = torch.from_numpy(embedding).to(device).unsqueeze(0)
        logits = head(q).squeeze(0)
    return _topk_from_logits(logits, ckpt["class_paths"], topk)


def predict_pil_image(pil_image: Image.Image, ckpt, embedder, head, device, topk: int = 5) -> list[dict]:
    """Return top-k predictions: each dict has class_path, display_name, probability."""
    emb = query_embedding_from_pil(pil_image, embedder)
    return predict_from_embedding(emb, ckpt, head, device, topk=topk)

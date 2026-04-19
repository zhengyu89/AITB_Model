"""
Train a small classifier on top of frozen DINOv2 embeddings (linear probe).
Produces a customer-deliverable checkpoint, e.g. my_landmark.pth.

You still use DINOv2 for features; only the final linear layer is your trained model.
The saved .pth is a classifier head checkpoint, not a standalone embedding model.

Run from repo root:  python scripts/train_landmark_head.py
  Attractions only (recommended if food/ confuses landscapes):
  python scripts/train_landmark_head.py --subset-prefix attraction
  Food only:
  python scripts/train_landmark_head.py --subset-prefix food
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
from tqdm import tqdm

from app.config import ATTRACTION_CHECKPOINT, FOOD_CHECKPOINT, SUPPORTED_IMAGE_EXTENSIONS, get_settings
from app.services.embedder import DinoV2Embedder


def collect_class_dirs(data_dir: Path) -> list[Path]:
    class_dirs = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_dir():
            continue
        has_images = any(
            child.is_file() and child.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            for child in path.iterdir()
        )
        if has_images:
            class_dirs.append(path)
    return class_dirs

def filter_class_dirs(class_dirs: list[Path], data_dir: Path, subset_prefix: str | None) -> list[Path]:
    if not subset_prefix:
        return class_dirs
    prefix = subset_prefix.strip("/").replace("\\", "/").rstrip("/")
    out: list[Path] = []
    for d in class_dirs:
        rel = d.relative_to(data_dir).as_posix()
        if rel == prefix or rel.startswith(prefix + "/"):
            out.append(d)
    return out


def collect_samples_for_class_dirs(
    data_dir: Path, class_dirs: list[Path]
) -> list[tuple[Path, str, str, int]]:
    class_paths = [str(d.relative_to(data_dir)) for d in class_dirs]
    path_to_idx = {p: i for i, p in enumerate(class_paths)}
    samples: list[tuple[Path, str, str, int]] = []
    for class_dir in class_dirs:
        class_path = str(class_dir.relative_to(data_dir))
        class_idx = path_to_idx[class_path]
        class_name = class_dir.name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                samples.append((image_path, class_name, class_path, class_idx))
    return samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a linear head on frozen DINOv2 embeddings; save a fixed checkpoint name."
    )
    p.add_argument("--data-dir", type=Path, default=_REPO_ROOT / "data" / "reference")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--subset-prefix",
        type=str,
        default=None,
        metavar="PREFIX",
        help="Only train on classes under this path prefix, e.g. 'attraction' excludes food/ from the same softmax.",
    )
    return p.parse_args()


def resolve_output_path(subset_prefix: str | None) -> Path:
    if subset_prefix is None:
        raise SystemExit("subset-prefix is required and must be either 'attraction' or 'food'.")

    normalized = subset_prefix.strip("/").replace("\\", "/").rstrip("/")
    if normalized == "attraction":
        return _REPO_ROOT / ATTRACTION_CHECKPOINT
    if normalized == "food":
        return _REPO_ROOT / FOOD_CHECKPOINT
    raise SystemExit("subset-prefix must be exactly 'attraction' or 'food'.")


def main() -> int:
    args = parse_args()
    settings = get_settings()
    embedding_model_name = settings.embedding_model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = resolve_output_path(args.subset_prefix)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    class_dirs = filter_class_dirs(collect_class_dirs(args.data_dir), args.data_dir, args.subset_prefix)
    if not class_dirs:
        raise SystemExit(
            f"No classes after filter under {args.data_dir}"
            + (f" (subset-prefix={args.subset_prefix!r})" if args.subset_prefix else "")
        )

    samples = collect_samples_for_class_dirs(args.data_dir, class_dirs)
    if not samples:
        raise SystemExit(f"No images in filtered classes under {args.data_dir}")

    class_paths = [str(d.relative_to(args.data_dir)) for d in class_dirs]
    num_classes = len(class_paths)

    indices = list(range(len(samples)))
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * args.val_ratio)) if len(indices) > 10 else 0
    val_idx = set(indices[:n_val]) if n_val else set()
    train_samples = [samples[i] for i in indices if i not in val_idx]
    val_samples = [samples[i] for i in indices if i in val_idx]

    embedder = DinoV2Embedder(model_name=embedding_model_name)
    for param in embedder.model.parameters():
        param.requires_grad = False

    head = nn.Linear(embedder.embedding_dim, num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    def run_epoch(batch_paths: list, batch_labels: list, train: bool) -> tuple[float, float]:
        if train:
            head.train()
        else:
            head.eval()
        total_loss = 0.0
        n = 0
        correct = 0
        order = list(range(len(batch_paths)))
        if train:
            random.shuffle(order)
        with torch.set_grad_enabled(train):
            for start in tqdm(range(0, len(order), args.batch_size), leave=False):
                chunk = order[start : start + args.batch_size]
                paths = [batch_paths[i] for i in chunk]
                labels = torch.tensor([batch_labels[i] for i in chunk], device=device)
                with torch.no_grad():
                    emb = torch.from_numpy(embedder.embed_paths(paths)).to(device)
                logits = head(emb)
                loss = criterion(logits, labels)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                total_loss += loss.item() * len(chunk)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                n += len(chunk)
        return total_loss / max(n, 1), correct / max(n, 1)

    train_paths = [str(s[0]) for s in train_samples]
    train_labels = [s[3] for s in train_samples]
    val_paths = [str(s[0]) for s in val_samples]
    val_labels = [s[3] for s in val_samples]

    for epoch in range(args.epochs):
        tr_loss, tr_acc = run_epoch(train_paths, train_labels, train=True)
        if val_paths:
            va_loss, va_acc = run_epoch(val_paths, val_labels, train=False)
            print(
                f"epoch {epoch + 1}/{args.epochs}  train loss={tr_loss:.4f} acc={tr_acc:.3f}  "
                f"val loss={va_loss:.4f} acc={va_acc:.3f}"
            )
        else:
            print(f"epoch {epoch + 1}/{args.epochs}  train loss={tr_loss:.4f} acc={tr_acc:.3f}")

    payload = {
        "head_state_dict": head.state_dict(),
        "class_paths": class_paths,
        "class_path_to_idx": {p: i for i, p in enumerate(class_paths)},
        "dinov2_model_name": embedding_model_name,
        "embedding_dim": embedder.embedding_dim,
        "num_classes": num_classes,
        "data_dir_relative": str(args.data_dir),
        "subset_prefix": args.subset_prefix,
    }
    torch.save(payload, out_path)
    print("Saved:", out_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

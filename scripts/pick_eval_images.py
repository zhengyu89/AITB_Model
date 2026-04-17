"""
Sample a few reference images per class and perturb them with albumentations
(rotation + lighting changes) to produce manual evaluation queries.

Outputs land in data/test/eval_picks/ alongside a manifest.json mapping each
generated file back to its expected class path, so Web UI checks can tell
whether the retrieval pipeline still recognizes a perturbed query.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

try:
    import albumentations as A
except ImportError as exc:
    raise SystemExit(
        "albumentations is not installed. Run `./venv/bin/pip install -r requirements.txt` first."
    ) from exc

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.services.embedder import SUPPORTED_EXTENSIONS

DEFAULT_INPUT_DIR = _REPO_ROOT / "data" / "reference"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "test" / "eval_picks"
DEFAULT_PER_CLASS = 3
DEFAULT_JPEG_QUALITY = 95
DEFAULT_VARIANTS = ("medium", "hard")


def make_gauss_noise() -> A.BasicTransform:
    params = inspect.signature(A.GaussNoise).parameters
    if "var_limit" in params:
        kwargs = {"var_limit": (20.0, 80.0), "p": 1.0}
        return A.GaussNoise(**kwargs)
    if "std_range" in params:
        kwargs = {"std_range": (0.04, 0.12), "mean_range": (0.0, 0.0), "p": 1.0}
        return A.GaussNoise(**kwargs)
    return A.GaussNoise(p=1.0)


def make_image_compression(low: int, high: int) -> A.BasicTransform:
    params = inspect.signature(A.ImageCompression).parameters
    if "quality_range" in params:
        kwargs = {"quality_range": (low, high), "p": 1.0}
        return A.ImageCompression(**kwargs)
    kwargs = {"quality_lower": low, "quality_upper": high, "p": 1.0}
    return A.ImageCompression(**kwargs)


def build_eval_transforms() -> dict[str, A.Compose]:
    return {
        "medium": A.Compose(
            [
                A.Affine(
                    rotate=(-18, 18),
                    scale=(0.95, 1.05),
                    translate_percent=(-0.03, 0.03),
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1.0,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.35, 0.35),
                    contrast_limit=(-0.15, 0.15),
                    p=1.0,
                ),
            ]
        ),
        "hard": A.Compose(
            [
                A.Affine(
                    rotate=(-28, 28),
                    scale=(0.88, 1.10),
                    shear=(-8, 8),
                    translate_percent=(-0.06, 0.06),
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1.0,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.45, 0.45),
                    contrast_limit=(-0.25, 0.25),
                    p=1.0,
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                make_image_compression(45, 75),
            ]
        ),
        "extreme": A.Compose(
            [
                A.Perspective(scale=(0.05, 0.12), keep_size=True, p=1.0),
                A.Affine(
                    rotate=(-40, 40),
                    scale=(0.80, 1.15),
                    shear=(-12, 12),
                    translate_percent=(-0.10, 0.10),
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1.0,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.55, 0.55),
                    contrast_limit=(-0.35, 0.35),
                    p=1.0,
                ),
                A.MotionBlur(blur_limit=(5, 9), p=1.0),
                make_gauss_noise(),
                make_image_compression(25, 55),
            ]
        ),
    }


def collect_class_dirs(root: Path) -> list[Path]:
    class_dirs: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        has_images = any(
            child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
            for child in path.iterdir()
        )
        if has_images:
            class_dirs.append(path)
    return class_dirs


def collect_class_images(class_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.ascontiguousarray(np.array(image))


def save_jpeg(array: np.ndarray, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")
    image.save(path, format="JPEG", quality=quality, optimize=True, subsampling=0)


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def make_output_path(
    src: Path,
    relative_class_dir: Path,
    output_dir: Path,
    reserved_paths: set[Path],
    variant_name: str,
) -> Path:
    output_path = output_dir / relative_class_dir / f"{src.stem}__{variant_name}.jpg"
    if output_path in reserved_paths:
        output_path = (
            output_dir
            / relative_class_dir
            / f"{src.stem}__from_{src.suffix.lower().lstrip('.')}__{variant_name}.jpg"
        )
    reserved_paths.add(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample reference images per class and augment them for manual eval."
    )
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--per-class",
        type=int,
        default=DEFAULT_PER_CLASS,
        help=f"How many source images to draw per leaf class (default: {DEFAULT_PER_CLASS}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed. Omit for a fresh random draw each run.",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        choices=list(DEFAULT_VARIANTS),
        help="Which evaluation difficulty variants to generate for each sampled image.",
    )
    p.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input_dir.exists():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1
    if args.per_class < 1:
        print("--per-class must be >= 1.", file=sys.stderr)
        return 1
    if not 1 <= args.jpeg_quality <= 100:
        print("--jpeg-quality must be between 1 and 100.", file=sys.stderr)
        return 1

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        set_seed = getattr(A, "set_seed", None)
        if callable(set_seed):
            set_seed(args.seed)

    class_dirs = collect_class_dirs(args.input_dir)
    if not class_dirs:
        print(f"No supported images found in {args.input_dir}", file=sys.stderr)
        return 1

    transforms = build_eval_transforms()
    variant_names = list(dict.fromkeys(args.variants))

    reset_output_dir(args.output_dir)
    manifest: list[dict] = []
    reserved_paths: set[Path] = set()
    sampled_source_count = 0

    for class_dir in class_dirs:
        relative_class_dir = class_dir.relative_to(args.input_dir)
        class_images = collect_class_images(class_dir)
        if not class_images:
            continue

        chosen = random.sample(class_images, min(args.per_class, len(class_images)))
        sampled_source_count += len(chosen)

        for src in chosen:
            image = load_image_rgb(src)
            class_path = str(relative_class_dir).replace("\\", "/")
            for variant_name in variant_names:
                augmented = transforms[variant_name](image=image)["image"]
                out_path = make_output_path(
                    src=src,
                    relative_class_dir=relative_class_dir,
                    output_dir=args.output_dir,
                    reserved_paths=reserved_paths,
                    variant_name=variant_name,
                )
                save_jpeg(augmented, out_path, quality=args.jpeg_quality)

                manifest.append(
                    {
                        "output": str(out_path.relative_to(_REPO_ROOT)),
                        "copied_to": str(out_path.relative_to(_REPO_ROOT)),
                        "source": str(src.relative_to(_REPO_ROOT)),
                        "expected_class_path": class_path,
                        "variant": variant_name,
                    }
                )

    meta_path = args.output_dir / "manifest.json"
    meta_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"Sampled {sampled_source_count} source image(s) from {len(class_dirs)} class(es); "
        f"generated {len(manifest)} eval query file(s) "
        f"across {len(variant_names)} variant(s) under {args.output_dir}"
    )
    print("Manifest:", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

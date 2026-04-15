from __future__ import annotations

import argparse
from collections import defaultdict
import inspect
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

try:
    import albumentations as A
except ImportError as exc:
    raise SystemExit(
        "albumentations is not installed. Run `./venv/bin/pip install -r requirements.txt` first."
    ) from exc


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_INPUT_DIR = Path("data/reference")
DEFAULT_OUTPUT_DIR = Path("data/train")
DEFAULT_VARIANTS = ("clean", "noise", "compressed")
ALPHA_BACKGROUND = (255, 255, 255, 255)
DEFAULT_MIN_CLASS_OUTPUTS = 80
DEFAULT_MAX_CLASS_OUTPUTS = 150
DEFAULT_LARGE_CLASS_THRESHOLD = 100
DEFAULT_LARGE_CLASS_SAMPLE_SIZE = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment images from data/reference and save all outputs as JPEG."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Source image root. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Destination root. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved outputs. Default: 95",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N source images. Useful for a quick smoke test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible augmentations. Default: 42",
    )
    parser.add_argument(
        "--enable-horizontal-flip",
        action="store_true",
        help="Enable horizontal flip. Keep this off if mirrored text/logo is a concern.",
    )
    parser.add_argument(
        "--min-class-outputs",
        type=int,
        default=DEFAULT_MIN_CLASS_OUTPUTS,
        help="Minimum number of augmented JPEGs to generate per class. Default: 50",
    )
    parser.add_argument(
        "--max-class-outputs",
        type=int,
        default=DEFAULT_MAX_CLASS_OUTPUTS,
        help="Maximum number of augmented JPEGs to generate per class. Default: 80",
    )
    parser.add_argument(
        "--large-class-threshold",
        type=int,
        default=DEFAULT_LARGE_CLASS_THRESHOLD,
        help=(
            "If a class has more than this many source images, only a random subset will be "
            "used for augmentation. Default: 100"
        ),
    )
    parser.add_argument(
        "--large-class-sample-size",
        type=int,
        default=DEFAULT_LARGE_CLASS_SAMPLE_SIZE,
        help="Number of source images to sample for very large classes. Default: 15",
    )
    return parser.parse_args()


def build_pipelines(enable_horizontal_flip: bool) -> dict[str, A.Compose]:
    base_transforms = []
    if enable_horizontal_flip:
        base_transforms.append(A.HorizontalFlip(p=0.5))

    base_transforms.extend(
        [
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
        ]
    )

    return {
        "clean": A.Compose(list(base_transforms)),
        "noise": A.Compose(
            [
                *base_transforms,
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                make_gauss_noise(),
            ]
        ),
        "compressed": A.Compose(
            [
                *base_transforms,
                make_image_compression(),
            ]
        ),
    }


def make_gauss_noise():
    params = inspect.signature(A.GaussNoise).parameters
    if "var_limit" in params:
        return A.GaussNoise(var_limit=(10.0, 50.0), p=0.2)
    if "std_range" in params:
        return A.GaussNoise(std_range=(0.02, 0.08), mean_range=(0.0, 0.0), p=0.2)
    return A.GaussNoise(p=0.2)


def make_image_compression():
    params = inspect.signature(A.ImageCompression).parameters
    if "quality_range" in params:
        return A.ImageCompression(quality_range=(70, 100), p=0.3)
    return A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3)


def collect_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def group_images_by_directory(
    image_paths: list[Path], input_dir: Path
) -> dict[Path, list[Path]]:
    groups: dict[Path, list[Path]] = {}
    for image_path in image_paths:
        relative_dir = image_path.relative_to(input_dir).parent
        groups.setdefault(relative_dir, []).append(image_path)
    return groups


def resolve_target_output_count(
    original_count: int,
    min_class_outputs: int,
    max_class_outputs: int,
) -> int:
    return min(max_class_outputs, max(min_class_outputs, original_count))


def build_class_plans(
    images_by_directory: dict[Path, list[Path]],
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, object]]:
    class_plans: list[dict[str, object]] = []

    for class_dir in sorted(images_by_directory):
        source_images = sorted(images_by_directory[class_dir])
        original_count = len(source_images)
        sampled_images = list(source_images)

        if original_count > args.large_class_threshold:
            sampled_images = rng.sample(
                source_images,
                k=min(args.large_class_sample_size, original_count),
            )

        target_output_count = resolve_target_output_count(
            original_count=original_count,
            min_class_outputs=args.min_class_outputs,
            max_class_outputs=args.max_class_outputs,
        )

        repeat_counts: dict[tuple[Path, str], int] = defaultdict(int)
        class_jobs: list[tuple[Path, str, int]] = []
        source_cycle: list[Path] = []
        variant_cycle: list[str] = []

        while len(class_jobs) < target_output_count:
            if not source_cycle:
                source_cycle = rng.sample(sampled_images, len(sampled_images))
            if not variant_cycle:
                variant_cycle = rng.sample(list(DEFAULT_VARIANTS), len(DEFAULT_VARIANTS))

            source_image = source_cycle.pop()
            variant_name = variant_cycle.pop()
            repeat_counts[(source_image, variant_name)] += 1
            class_jobs.append(
                (source_image, variant_name, repeat_counts[(source_image, variant_name)])
            )

        class_plans.append(
            {
                "class_dir": class_dir,
                "original_count": original_count,
                "sampled_count": len(sampled_images),
                "target_output_count": target_output_count,
                "jobs": class_jobs,
            }
        )

    return class_plans


def load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)

        if image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        ):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, ALPHA_BACKGROUND)
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")

        return np.ascontiguousarray(np.array(image))


def make_output_path(
    image_path: Path,
    input_dir: Path,
    output_dir: Path,
    variant_name: str,
    repeat_index: int,
    reserved_paths: set[Path],
) -> Path:
    relative_dir = image_path.relative_to(input_dir).parent
    base_name = f"{image_path.stem}__{variant_name}_{repeat_index:02d}.jpg"
    output_path = output_dir / relative_dir / base_name

    if output_path in reserved_paths:
        output_path = (
            output_dir
            / relative_dir
            / (
                f"{image_path.stem}__from_{image_path.suffix.lower().lstrip('.')}__"
                f"{variant_name}_{repeat_index:02d}.jpg"
            )
        )

    reserved_paths.add(output_path)
    return output_path


def save_as_jpeg(image_array: np.ndarray, output_path: Path, quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8), mode="RGB")
    image.save(output_path, format="JPEG", quality=quality, optimize=True, subsampling=0)


def main() -> int:
    args = parse_args()

    if not args.input_dir.exists():
        print(f"Input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 1

    if not 1 <= args.jpeg_quality <= 100:
        print("--jpeg-quality must be between 1 and 100.", file=sys.stderr)
        return 1
    if args.min_class_outputs < 1:
        print("--min-class-outputs must be at least 1.", file=sys.stderr)
        return 1
    if args.max_class_outputs < args.min_class_outputs:
        print("--max-class-outputs must be >= --min-class-outputs.", file=sys.stderr)
        return 1
    if args.large_class_sample_size < 1:
        print("--large-class-sample-size must be at least 1.", file=sys.stderr)
        return 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    if hasattr(A, "set_seed"):
        A.set_seed(args.seed)
    rng = random.Random(args.seed)

    all_image_paths = collect_images(args.input_dir)
    image_paths = all_image_paths
    if args.limit is not None:
        image_paths = all_image_paths[: args.limit]

    if not image_paths:
        print(f"No supported images found in {args.input_dir}")
        return 0

    pipelines = build_pipelines(enable_horizontal_flip=args.enable_horizontal_flip)
    images_by_directory = group_images_by_directory(image_paths, args.input_dir)
    class_plans = build_class_plans(images_by_directory, args, rng)
    reserved_paths: set[Path] = set()
    failed_jobs: list[tuple[Path, str, str]] = []
    generated_count = 0

    total_jobs = sum(len(class_plan["jobs"]) for class_plan in class_plans)

    with tqdm(total=total_jobs, desc="Augmenting", unit="image") as progress:
        for class_plan in class_plans:
            class_dir = class_plan["class_dir"]
            class_jobs = class_plan["jobs"]
            class_cache: dict[Path, np.ndarray] = {}

            for image_path, variant_name, repeat_index in class_jobs:
                try:
                    if image_path not in class_cache:
                        class_cache[image_path] = load_image_rgb(image_path)

                    image = class_cache[image_path]
                    augmented = pipelines[variant_name](image=image)["image"]
                    output_path = make_output_path(
                        image_path=image_path,
                        input_dir=args.input_dir,
                        output_dir=args.output_dir,
                        variant_name=variant_name,
                        repeat_index=repeat_index,
                        reserved_paths=reserved_paths,
                    )
                    save_as_jpeg(augmented, output_path, quality=args.jpeg_quality)
                    generated_count += 1
                except Exception as exc:
                    failed_jobs.append((image_path, str(class_dir), str(exc)))
                finally:
                    progress.update(1)

    sampled_large_classes = [
        class_plan
        for class_plan in class_plans
        if class_plan["original_count"] > args.large_class_threshold
    ]
    print(
        f"Planned {len(class_plans)} classes from {len(image_paths)} source images."
    )
    print(f"Generated {generated_count} JPEG files in {args.output_dir}")
    print(
        f"Large classes sampled down to {args.large_class_sample_size} source images: "
        f"{len(sampled_large_classes)}"
    )

    if failed_jobs:
        print("\nFailed jobs:", file=sys.stderr)
        for image_path, class_dir, message in failed_jobs[:20]:
            print(f"- {class_dir} :: {image_path}: {message}", file=sys.stderr)
        if len(failed_jobs) > 20:
            print(
                f"... and {len(failed_jobs) - 20} more failures.",
                file=sys.stderr,
            )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

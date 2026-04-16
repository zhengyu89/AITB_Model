"""
Copy a few reference images into data/test/eval_picks/ for quick Web UI testing.

Does not download from the internet — only samples from your data/reference tree.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.services.embedder import SUPPORTED_EXTENSIONS

EVAL_OUT_DIR = _REPO_ROOT / "data" / "test" / "eval_picks"


def collect_class_dirs(data_dir: Path) -> list[Path]:
    class_dirs = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_dir():
            continue
        has_images = any(
            child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
            for child in path.iterdir()
        )
        if has_images:
            class_dirs.append(path)
    return class_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample images from data/reference for manual evaluation.")
    p.add_argument("--data-dir", type=Path, default=_REPO_ROOT / "data" / "reference")
    p.add_argument(
        "--per-class",
        type=int,
        default=1,
        help="Randomly copy this many images per leaf class (default 1).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for class_dir in collect_class_dirs(args.data_dir):
        rel = class_dir.relative_to(args.data_dir)
        files = [
            p
            for p in sorted(class_dir.iterdir())
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not files:
            continue
        k = min(args.per_class, len(files))
        chosen = random.sample(files, k) if k < len(files) else list(files)
        for src in chosen:
            dest_dir = EVAL_OUT_DIR / rel
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            shutil.copy2(src, dest)
            manifest.append(
                {
                    "copied_to": str(dest.relative_to(_REPO_ROOT)),
                    "expected_class_path": str(rel).replace("\\", "/"),
                    "source": str(src.relative_to(_REPO_ROOT)),
                }
            )

    meta_path = EVAL_OUT_DIR / "manifest.json"
    meta_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Copied {len(manifest)} file(s) under {EVAL_OUT_DIR}")
    print("Manifest:", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

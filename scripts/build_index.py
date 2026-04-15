import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from dinov2_embedder import DEFAULT_MODEL_NAME, SUPPORTED_EXTENSIONS, DinoV2Embedder


DEFAULT_DATA_DIR = Path("data/reference")
DEFAULT_INDEX_OUT = Path("faiss_index_dinov2.ivf")
DEFAULT_EMB_OUT = Path("reference_embeddings_dinov2.npy")
DEFAULT_META_OUT = Path("reference_meta_dinov2.json")
DEFAULT_BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index from DINOv2 embeddings.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--index-out", type=Path, default=DEFAULT_INDEX_OUT)
    parser.add_argument("--emb-out", type=Path, default=DEFAULT_EMB_OUT)
    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META_OUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


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


def collect_reference_images(data_dir: Path) -> list[tuple[Path, str, str, int]]:
    samples: list[tuple[Path, str, str, int]] = []

    for class_idx, class_dir in enumerate(collect_class_dirs(data_dir)):
        class_name = class_dir.name
        class_path = str(class_dir.relative_to(data_dir))
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                samples.append((image_path, class_name, class_path, class_idx))

    return samples


def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    n_vectors = len(embeddings)
    if n_vectors < 20000:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index, "flat_exact"

    nlist = max(1, min(100, int(np.sqrt(n_vectors))))
    if nlist == 1:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index, "flat_exact"

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    return index, f"ivf_nlist={nlist}"


def main() -> int:
    args = parse_args()
    samples = collect_reference_images(args.data_dir)
    if not samples:
        raise SystemExit(f"No supported images found in {args.data_dir}")

    embedder = DinoV2Embedder(model_name=args.model_name, device=args.device)

    embeddings = []
    for start in tqdm(range(0, len(samples), args.batch_size), desc="Embedding", unit="batch"):
        batch = samples[start : start + args.batch_size]
        embeddings.append(embedder.embed_paths([path for path, _, _, _ in batch]))

    emb_array = np.vstack(embeddings).astype("float32")
    np.save(args.emb_out, emb_array)

    meta = [
        {
            "path": str(path),
            "class_name": class_name,
            "class_path": class_path,
            "class_idx": class_idx,
        }
        for path, class_name, class_path, class_idx in samples
    ]
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    index, index_mode = build_faiss_index(emb_array)
    faiss.write_index(index, str(args.index_out))
    print("Saved index:", args.index_out, "emb shape:", emb_array.shape, "mode:", index_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

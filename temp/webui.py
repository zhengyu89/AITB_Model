"""
Streamlit UI: attraction and food linear heads with Qdrant retrieval.

From repo root:
  ./venv/bin/streamlit run temp/webui.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.services.classifier import (
    load_head_branch,
    load_landmark_classifier,
    predict_from_embedding,
    query_embedding_from_pil,
)
from app.services.qdrant_retrieval import aggregate_qdrant_results, qdrant_topk
from app.config import ATTRACTION_CHECKPOINT, FOOD_CHECKPOINT, SUPPORTED_IMAGE_EXTENSIONS, get_settings
from qdrant_client import QdrantClient


EVAL_PICKS_DIR = _REPO_ROOT / "data" / "test" / "eval_picks"


def _resolve_file(user_path: str, repo_root: Path) -> Path | None:
    path = Path(user_path).expanduser()
    if path.is_file():
        return path
    alt = repo_root / user_path
    if alt.is_file():
        return alt
    return None


def _read_checkpoint_meta(path: Path, device: str) -> dict:
    ckpt = torch.load(path, map_location=device)
    return {
        "path": path,
        "dinov2_model_name": str(ckpt.get("dinov2_model_name") or ""),
        "embedding_dim": int(ckpt.get("embedding_dim") or 0),
    }


def _validate_checkpoint_family(ckpts: dict[str, Path], device: str) -> None:
    metas = {name: _read_checkpoint_meta(path, device) for name, path in ckpts.items()}
    models = {meta["dinov2_model_name"] for meta in metas.values()}
    dims = {meta["embedding_dim"] for meta in metas.values()}
    if len(models) <= 1 and len(dims) <= 1:
        return

    lines = ["Detected checkpoints built with different DINO backbones. The Web UI cannot mix them:"]
    for name, meta in metas.items():
        lines.append(
            f"- {name}: `{meta['path'].name}` -> {meta['dinov2_model_name']} / dim={meta['embedding_dim']}"
        )
    lines.append("Use the same DINO backbone for attraction, food, and the Qdrant index.")
    raise ValueError("\n".join(lines))


def _bucket_zh(bucket: str | None) -> str:
    if bucket == "food":
        return "Food"
    if bucket == "attraction":
        return "Attraction"
    return "Unknown"


def _decide_final(q_groups: list[dict], accept_score: float, tentative_score: float, min_gap: float) -> dict:
    if q_groups:
        top = q_groups[0]
        score = float(top["best_score"])
        second_score = float(q_groups[1]["best_score"]) if len(q_groups) > 1 else 0.0
        gap = score - second_score
        bucket = str(top.get("category") or "")

        if score >= accept_score and gap >= min_gap:
            return {
                "status": "accept",
                "bucket": bucket,
                "display_name": str(top["display_name"]),
                "class_path": str(top["class_path"]),
                "score": score,
                "gap": gap,
                "hit_count": int(top["hit_count"]),
                "candidates": q_groups[:3],
                "detail": (
                    f"Qdrant grouped Top-1 **{score:.4f}** >= accept threshold **{accept_score:.2f}** "
                    f"and the gap to the second result **{gap:.4f}** >= minimum gap **{min_gap:.2f}**."
                ),
            }

        if score >= tentative_score:
            reasons: list[str] = []
            if score < accept_score:
                reasons.append(f"Top-1 **{score:.4f}** is below the accept threshold **{accept_score:.2f}**")
            if gap < min_gap:
                reasons.append(f"The gap to the second result **{gap:.4f}** is below the minimum gap **{min_gap:.2f}**")
            if not reasons:
                reasons.append("The retrieval result still needs manual confirmation")
            return {
                "status": "tentative",
                "bucket": bucket,
                "display_name": str(top["display_name"]),
                "class_path": str(top["class_path"]),
                "score": score,
                "gap": gap,
                "hit_count": int(top["hit_count"]),
                "candidates": q_groups[:3],
                "detail": "; ".join(reasons) + ". Showing candidates only, without auto-confirmation.",
            }

    return {
        "status": "reject",
        "bucket": None,
        "display_name": "",
        "class_path": "",
        "score": None,
        "gap": None,
        "hit_count": 0,
        "candidates": [],
        "detail": "Qdrant returned no result, or the overall similarity is too low to confirm a specific place.",
    }


@st.cache_resource
def _load_inference_bundle(attr_resolved: str, food_resolved: str, dev: str):
    ap = Path(attr_resolved) if attr_resolved else None
    fp = Path(food_resolved) if food_resolved else None
    if (not ap or not ap.is_file()) and (not fp or not fp.is_file()):
        raise ValueError("At least one checkpoint is required: attraction or food.")

    ckpts: dict[str, Path] = {}
    if ap and ap.is_file():
        ckpts["attraction"] = ap
    if fp and fp.is_file():
        ckpts["food"] = fp
    _validate_checkpoint_family(ckpts, dev)

    base = ap if ap and ap.is_file() else fp
    assert base is not None
    ck0, embedder, head0, device = load_landmark_classifier(str(base.resolve()), dev)
    branches: dict[str, tuple] = {}

    if ap and ap.is_file():
        if ap.resolve() == base.resolve():
            branches["attraction"] = (ck0, head0)
        else:
            ck, hd = load_head_branch(ap, embedder, device)
            branches["attraction"] = (ck, hd)

    if fp and fp.is_file():
        if fp.resolve() == base.resolve():
            branches["food"] = (ck0, head0)
        else:
            ck, hd = load_head_branch(fp, embedder, device)
            branches["food"] = (ck, hd)

    return embedder, branches, device


@st.cache_resource
def _qdrant_client(url: str, api_key: str | None) -> QdrantClient:
    return QdrantClient(url=url, api_key=api_key)


def _format_cls_rows(rows: list[dict]) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    raw["probability"] = raw["probability"].map(lambda x: f"{x:.2%}")
    raw = raw.rename(
        columns={
            "class_path": "Class Path",
            "display_name": "Display Name",
            "probability": "Probability",
        }
    )
    return pd.DataFrame(raw[["Display Name", "Class Path", "Probability"]])


def _render_cls_block(title: str, rows: list[dict] | None, missing_hint: str) -> None:
    st.subheader(title)
    if rows is None:
        st.caption(missing_hint)
        return
    st.dataframe(_format_cls_rows(rows), use_container_width=True, hide_index=True)
    top = rows[0]
    st.success(f"**Top-1**: {top['display_name']} · **{top['probability']:.1%}** · `{top['class_path']}`")


def _candidate_table(rows: list[dict]) -> pd.DataFrame:
    data = []
    for row in rows:
        data.append(
            {
                "Name": row["display_name"],
                "Type": _bucket_zh(row.get("category")),
                "Similarity": f"{float(row['best_score']):.4f}",
            }
        )
    return pd.DataFrame(data)


def _decision_label(status: str) -> str:
    if status == "accept":
        return "High Confidence"
    if status == "tentative":
        return "Needs Review"
    return "No Reliable Match"


@st.cache_data
def _list_demo_images() -> list[Path]:
    if not EVAL_PICKS_DIR.exists():
        return []
    return sorted(
        path for path in EVAL_PICKS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def _demo_image_label(path: Path) -> str:
    rel = path.relative_to(EVAL_PICKS_DIR)
    parts = rel.parts
    if len(parts) >= 3:
        return f"{parts[-2]}/{parts[-1]}"
    return str(rel)


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="Malaysia Landmark Attraction & Food Recognition", layout="wide", initial_sidebar_state="collapsed")
    st.title("Malaysia Landmark Attraction & Food Recognition")

    with st.sidebar:
        device_options = ["cuda", "cpu"]
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = st.selectbox("Device", options=device_options, index=device_options.index(default_device))
        if dev == "cuda" and not torch.cuda.is_available():
            st.warning("CUDA is not available. Falling back to CPU.")
            dev = "cpu"

        topk = st.slider("Top-K per block", min_value=1, max_value=20, value=settings.default_topk)
        preview_px = st.slider("Preview width (px)", min_value=180, max_value=480, value=420, step=20)

        st.divider()
        st.markdown("**Classifier Checkpoints**")
        attr_ck = st.text_input(
            "Attraction .pth",
            value=str(_REPO_ROOT / ATTRACTION_CHECKPOINT),
        )
        food_ck = st.text_input(
            "Food .pth",
            value=str(_REPO_ROOT / FOOD_CHECKPOINT),
        )

        st.divider()
        st.markdown("**Qdrant**")
        use_qdrant = st.checkbox("Enable Qdrant retrieval", value=True)
        q_url = st.text_input("Qdrant URL", value=settings.qdrant_url)
        q_api_key = st.text_input("Qdrant API Key", value=settings.qdrant_api_key or "", type="password")
        q_coll = st.text_input("Collection", value=settings.qdrant_collection)
        accept_score = st.slider("Accept threshold", min_value=0.20, max_value=0.95, value=settings.accept_score, step=0.05)
        tentative_score = st.slider("Tentative threshold", min_value=0.10, max_value=0.80, value=settings.tentative_score, step=0.02)
        min_gap = st.slider("Minimum Top-1 / Top-2 gap", min_value=0.00, max_value=0.20, value=settings.min_gap, step=0.01)

    if tentative_score >= accept_score:
        tentative_score = max(0.0, accept_score - 0.01)
        st.warning("The tentative threshold must be lower than the accept threshold. It has been adjusted automatically.")

    apath = _resolve_file(attr_ck, _REPO_ROOT)
    fpath = _resolve_file(food_ck, _REPO_ROOT)
    if not apath and not fpath:
        st.error("No attraction or food checkpoint was found. At least one .pth file is required.")
        st.code(
            "python scripts/train_landmark_head.py --subset-prefix attraction\n"
            "python scripts/train_landmark_head.py --subset-prefix food",
            language="bash",
        )
        st.stop()

    try:
        embedder, branches, device = _load_inference_bundle(
            str(apath.resolve()) if apath else "",
            str(fpath.resolve()) if fpath else "",
            dev,
        )
    except Exception as exc:
        st.exception(exc)
        st.stop()

    # st.caption(f"DINOv2: `{embedder.model_name}` · Device: **{device}**")

    qdrant_client = None
    if use_qdrant:
        try:
            qdrant_client = _qdrant_client(q_url.strip(), q_api_key.strip() or None)
        except Exception as exc:
            st.warning(f"Failed to connect to Qdrant: {exc}")

    input_mode = st.radio(
        "Image Source",
        options=["Upload Image", "Choose Demo Image"],
        horizontal=True,
        index = 1
    )

    img: Image.Image | None = None
    image_caption = "Query Image"

    if input_mode == "Upload Image":
        up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
        if up is not None:
            img = Image.open(up)
            image_caption = "Uploaded Image"
    else:
        demo_images = _list_demo_images()
        if not demo_images:
            st.warning(f"No demo images were found under `{EVAL_PICKS_DIR.relative_to(_REPO_ROOT)}`.")
        else:
            selected_demo = st.selectbox(
                "Choose a demo image",
                options=demo_images,
                format_func=_demo_image_label,
            )
            img = Image.open(selected_demo)
            image_caption = _demo_image_label(selected_demo)

    if img is None:
        st.info("Choose a demo image or upload one. The page will then show classification, grouped Qdrant candidates, and the final decision.")
        return

    with st.spinner("Running inference..."):
        emb = query_embedding_from_pil(img, embedder)
        emb_list = emb.tolist()

        attr_rows = None
        if "attraction" in branches:
            ck, hd = branches["attraction"]
            attr_rows = predict_from_embedding(emb, ck, hd, device, topk=topk)

        food_rows = None
        if "food" in branches:
            ck, hd = branches["food"]
            food_rows = predict_from_embedding(emb, ck, hd, device, topk=topk)

        q_groups: list[dict] = []
        if qdrant_client is not None:
            try:
                q_rows = qdrant_topk(qdrant_client, q_coll.strip(), emb_list, topk)
                q_groups = aggregate_qdrant_results(q_rows)
            except Exception as exc:
                st.warning(f"Qdrant query failed: {exc}")

        final = _decide_final(q_groups, accept_score, tentative_score, min_gap)

    top_left, top_right = st.columns([1, 2])
    with top_left:
        st.image(img, caption=image_caption, width=int(preview_px))
    with top_right:
        st.markdown("### Final Decision")
        if final["status"] == "reject":
            st.error("No reliable match was found.")
            st.caption(final["detail"])
        elif final["status"] == "tentative":
            st.warning(
                f"**Possible match:** {final['display_name']}  \n"
                f"**Type:** {_bucket_zh(final['bucket'])}  \n"
                f"**Confidence:** {_decision_label(final['status'])}"
            )
            st.caption(final["detail"])
        else:
            st.success(
                f"**Match:** {final['display_name']}  \n"
                f"**Type:** {_bucket_zh(final['bucket'])}  \n"
                f"**Confidence:** {_decision_label(final['status'])}"
            )
            st.caption(final["detail"])

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Confidence", _decision_label(final["status"]))
        with m2:
            st.metric("Similarity", "-" if final["score"] is None else f"{final['score']:.4f}")
        with m3:
            st.metric("Reference Hits", int(final["hit_count"]))

        if final["candidates"]:
            st.markdown("#### Top Candidates")
            st.dataframe(
                _candidate_table(final["candidates"][:3]),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Technical Details", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            _render_cls_block(
                "Classification · Attraction",
                attr_rows,
                "Attraction checkpoint not found. Train one with `python scripts/train_landmark_head.py --subset-prefix attraction`.",
            )
        with c2:
            _render_cls_block(
                "Classification · Food",
                food_rows,
                "Food checkpoint not found. Train one with `python scripts/train_landmark_head.py --subset-prefix food`.",
            )
        with c3:
            st.subheader("Retrieval · Qdrant")
            if not use_qdrant or qdrant_client is None:
                st.caption("Qdrant is disabled or unavailable.")
            elif not q_groups:
                st.caption("No results were returned. The collection may be empty or not ingested yet.")
            else:
                df = pd.DataFrame(q_groups)
                df["best_score"] = df["best_score"].map(lambda x: f"{x:.4f}")
                df["total_score"] = df["total_score"].map(lambda x: f"{x:.4f}")
                df["category"] = df["category"].map(_bucket_zh)
                df = df.rename(
                    columns={
                        "display_name": "Display Name",
                        "class_path": "Class Path",
                        "category": "Bucket",
                        "best_score": "Best Similarity",
                        "total_score": "Total Similarity",
                        "hit_count": "Hit Count",
                    }
                )
                cols = [c for c in ["Display Name", "Bucket", "Class Path", "Best Similarity", "Total Similarity", "Hit Count"] if c in df.columns]
                st.dataframe(df[cols], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

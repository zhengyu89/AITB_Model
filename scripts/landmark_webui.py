"""
Streamlit UI: 景点 / 食物分类 + Qdrant 检索 +「最终确认」（Qdrant 优先，router 兜底）。

From repo root:
  ./venv/bin/streamlit run scripts/landmark_webui.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from qdrant_client import QdrantClient

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from landmark_inference import (
    load_head_branch,
    load_landmark_classifier,
    predict_from_embedding,
    query_embedding_from_pil,
)
from qdrant_retrieval import qdrant_topk

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "malaysia_landmarks_dinov2"


def _resolve_file(user_path: str, repo_root: Path) -> Path | None:
    path = Path(user_path).expanduser()
    if path.is_file():
        return path
    alt = repo_root / user_path
    if alt.is_file():
        return alt
    return None


def _bucket_from_qdrant_payload(class_path: str | None, category: str | None) -> str:
    cp = (class_path or "").replace("\\", "/")
    cat = (category or "").strip().lower()
    if cat == "food" or cp.startswith("food/"):
        return "food"
    return "attraction"


def _decide_final(
    emb: np.ndarray,
    q_rows: list[dict],
    attr_rows: list[dict] | None,
    food_rows: list[dict] | None,
    router_branch: tuple | None,
    device: str,
    qdrant_min_score: float,
) -> dict:
    """
    1) Qdrant Top-1 相似度 ≥ 阈值 → 最终 = 该点的 display / class_path（主路由）。
    2) 否则若有 router → 路由 attraction|food，再取对应细类头 Top-1。
    3) 否则无法自动确认。
    """
    if q_rows and float(q_rows[0].get("score") or 0) >= qdrant_min_score:
        top = q_rows[0]
        bucket = _bucket_from_qdrant_payload(top.get("class_path"), top.get("category"))
        return {
            "source": "qdrant",
            "bucket": bucket,
            "display_name": str(top.get("display_name") or ""),
            "class_path": str(top.get("class_path") or ""),
            "score": float(top.get("score") or 0),
            "detail": (
                f"规则：**Qdrant 优先**。Top-1 相似度 **{float(top.get('score') or 0):.4f}** ≥ 阈值 **{qdrant_min_score}** → "
                f"采用该参考点的名称与路径（大类 **{bucket}**）。"
            ),
        }

    if router_branch is not None:
        ck_r, hd_r = router_branch
        rrows = predict_from_embedding(emb, ck_r, hd_r, device, topk=2)
        rb = rrows[0]["class_path"]
        prob = float(rrows[0]["probability"])
        spec = None
        if rb == "food" and food_rows:
            spec = food_rows[0]
        elif rb == "attraction" and attr_rows:
            spec = attr_rows[0]
        if spec is not None:
            return {
                "source": "router",
                "bucket": rb,
                "display_name": spec["display_name"],
                "class_path": spec["class_path"],
                "score": None,
                "detail": (
                    f"规则：**Qdrant 无结果或低于阈值** → **景/食路由** 判为 **{rb}**（置信 **{prob:.1%}**），"
                    f"再取对应细类头 **Top-1**。"
                ),
            }
        return {
            "source": "router_coarse",
            "bucket": rb,
            "display_name": rb.replace("_", " ").title(),
            "class_path": rb,
            "score": None,
            "detail": f"规则：仅路由为 **{rb}**（**{prob:.1%}**），未加载对应细类权重。",
        }

    return {
        "source": "none",
        "bucket": None,
        "display_name": "",
        "class_path": "",
        "score": None,
        "detail": "Qdrant 不可用或 Top-1 低于阈值，且未配置 `my_landmark_router.pth`，无法自动给出最终结论。",
    }


@st.cache_resource
def _load_inference_bundle(attr_resolved: str, food_resolved: str, router_resolved: str, dev: str):
    ap = Path(attr_resolved) if attr_resolved else None
    fp = Path(food_resolved) if food_resolved else None
    if (not ap or not ap.is_file()) and (not fp or not fp.is_file()):
        raise ValueError("至少需要景点或食物权重之一")

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

    router_branch = None
    if router_resolved:
        rp = Path(router_resolved)
        if rp.is_file():
            ck_r, hd_r = load_head_branch(rp, embedder, device)
            router_branch = (ck_r, hd_r)

    return embedder, branches, router_branch, device


@st.cache_resource
def _qdrant_client(url: str) -> QdrantClient:
    return QdrantClient(url=url)


def _format_cls_rows(rows: list[dict]) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    raw["probability"] = raw["probability"].map(lambda x: f"{x:.2%}")
    raw = raw.rename(
        columns={
            "class_path": "文件夹路径",
            "display_name": "显示名",
            "probability": "概率",
        }
    )
    return pd.DataFrame(raw[["显示名", "文件夹路径", "概率"]])


def _render_cls_block(title: str, rows: list[dict] | None, missing_hint: str) -> None:
    st.subheader(title)
    if rows is None:
        st.caption(missing_hint)
        return
    st.dataframe(_format_cls_rows(rows), use_container_width=True, hide_index=True)
    b = rows[0]
    st.success(f"**Top-1**：{b['display_name']} · **{b['probability']:.1%}** · `{b['class_path']}`")


def main() -> None:
    st.set_page_config(page_title="Landmark demo", layout="wide")
    st.title("景点 + 食物 分类 + Qdrant + 最终确认")

    with st.expander("说明：两列 softmax、最终确认怎么来的", expanded=False):
        st.markdown(
            """
- **分类两列**仍是各自 softmax，**禁止跨列比概率**。
- **最终确认**（简单规则）：
  1. **Qdrant Top-1 相似度 ≥ 侧边栏阈值** → 直接采用该点的名称与路径（主路由，看 payload 的 `category` / `class_path` 定大类）。
  2. 否则若存在 **`my_landmark_router.pth`** → 先 **景/食二分类**，再取 **对应细类头 Top-1**。
  3. 否则提示无法自动确认。
- **路由训练**：`python scripts/train_landmark_head.py --router --out my_landmark_router.pth`
            """
        )

    with st.sidebar:
        dev = st.selectbox("Device", options=["cuda", "cpu"], index=0)
        if dev == "cuda":
            import torch

            if not torch.cuda.is_available():
                st.warning("CUDA 不可用，将使用 CPU。")
                dev = "cpu"
        topk = st.slider("每栏 Top-K", min_value=1, max_value=20, value=5)
        preview_px = st.slider(
            "查询图预览宽度 (px，固定不占满屏)",
            min_value=200,
            max_value=720,
            value=380,
            step=20,
        )

        st.divider()
        st.markdown("**分类权重**")
        attr_ck = st.text_input("景点 .pth", value=str(_REPO_ROOT / "my_landmark_attraction.pth"))
        food_ck = st.text_input("食物 .pth", value=str(_REPO_ROOT / "my_landmark_food.pth"))
        router_ck = st.text_input(
            "景/食路由 .pth（兜底）",
            value=str(_REPO_ROOT / "my_landmark_router.pth"),
        )
        st.caption("仅全类时可先把 `my_landmark.pth` 填在景点或食物其一。")

        st.divider()
        st.markdown("**Qdrant**")
        use_qdrant = st.checkbox("启用 Qdrant 检索", value=True)
        q_url = st.text_input("Qdrant URL", value=DEFAULT_QDRANT_URL)
        q_coll = st.text_input("Collection", value=DEFAULT_COLLECTION)
        q_min = st.slider("最终确认：Qdrant 最低相似度", min_value=0.15, max_value=0.95, value=0.35, step=0.05)

    apath = _resolve_file(attr_ck, _REPO_ROOT)
    fpath = _resolve_file(food_ck, _REPO_ROOT)
    rpath = _resolve_file(router_ck, _REPO_ROOT)

    if not apath and not fpath:
        st.error("找不到景点或食物权重，请至少训练并放置其中一个 .pth。")
        st.code(
            "python scripts/train_landmark_head.py --subset-prefix attraction --out my_landmark_attraction.pth\n"
            "python scripts/train_landmark_head.py --subset-prefix food --out my_landmark_food.pth\n"
            "python scripts/train_landmark_head.py --router --out my_landmark_router.pth",
            language="bash",
        )
        st.stop()

    try:
        embedder, branches, router_branch, device = _load_inference_bundle(
            str(apath.resolve()) if apath else "",
            str(fpath.resolve()) if fpath else "",
            str(rpath.resolve()) if rpath else "",
            dev,
        )
    except Exception as exc:
        st.exception(exc)
        st.stop()

    st.caption(f"DINOv2: `{embedder.model_name}` · 设备: **{device}**")
    if router_branch is None:
        st.caption("未检测到路由权重：Qdrant 低于阈值时将无法自动兜底。")

    qdrant_client = None
    if use_qdrant:
        try:
            qdrant_client = _qdrant_client(q_url.strip())
        except Exception as exc:
            st.warning(f"连接 Qdrant 失败: {exc}")

    up = st.file_uploader("上传照片", type=["jpg", "jpeg", "png", "webp"])
    if up is None:
        st.info("上传后显示：细类两列 + Qdrant + **最终确认**。")
        return

    from PIL import Image

    img = Image.open(up)
    with st.spinner("推理中…"):
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

        q_rows: list[dict] = []
        if qdrant_client is not None:
            try:
                q_rows = qdrant_topk(qdrant_client, q_coll.strip(), emb_list, topk)
            except Exception as exc:
                st.warning(f"Qdrant 查询失败: {exc}")

        final = _decide_final(emb, q_rows, attr_rows, food_rows, router_branch, device, q_min)

    st.image(img, caption="查询图", width=int(preview_px))

    st.markdown("### 最终确认")
    if final["source"] == "none":
        st.error(final["detail"])
    else:
        bucket_zh = "食物" if final["bucket"] == "food" else "景点"
        score_line = (
            f" · 相似度 **{final['score']:.4f}**" if final.get("score") is not None else ""
        )
        st.success(
            f"**{final['display_name']}** · `{final['class_path']}` · 大类：**{bucket_zh}**{score_line}\n\n"
            f"_来源：{'Qdrant' if final['source'] == 'qdrant' else '路由兜底'}_ · {final['detail']}"
        )

    # if attr_rows is not None and food_rows is not None:
    #     st.warning(
    #         "**不要**比较「食物列 vs 景点列」的 softmax 数值。**最终确认**只看上方规则（Qdrant → 路由），不是两列里挑概率大的。"
    #     )

    c1, c2, c3 = st.columns(3)

    with c1:
        _render_cls_block(
            "分类 · 景点",
            attr_rows,
            "未找到景点权重。训练：`python scripts/train_landmark_head.py --subset-prefix attraction --out my_landmark_attraction.pth`",
        )

    with c2:
        _render_cls_block(
            "分类 · 食物",
            food_rows,
            "未找到食物权重。训练：`python scripts/train_landmark_head.py --subset-prefix food --out my_landmark_food.pth`",
        )

    with c3:
        st.subheader("检索 · Qdrant")
        if not use_qdrant or qdrant_client is None:
            st.caption("未启用 Qdrant 或连接失败。")
        elif not q_rows:
            st.caption("无结果（集合为空或未 ingest）。")
        else:
            dfq = pd.DataFrame(q_rows)
            dfq["score"] = dfq["score"].map(lambda x: f"{x:.4f}")
            dfq = dfq.rename(
                columns={
                    "display_name": "显示名",
                    "class_path": "文件夹路径",
                    "category": "大类",
                    "image_path": "参考图",
                    "score": "相似度",
                }
            )
            cols = [c for c in ["显示名", "大类", "文件夹路径", "相似度", "参考图"] if c in dfq.columns]
            st.dataframe(dfq[cols], use_container_width=True, hide_index=True)
            r = q_rows[0]
            st.info(
                f"**检索最近**：{r.get('display_name')} · **{r.get('score', 0):.4f}** · `{r.get('class_path')}`"
            )


if __name__ == "__main__":
    main()

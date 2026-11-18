"""Streamlit Community Cloud entry point for the CRDIC POC."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(os.environ.get("CRDIC_BUNDLE_ROOT", Path(__file__).resolve().parent))
ASSETS = ROOT / "assets"

try:
    from model.demo_gallery import DemoGallery
    from model.metrics import lpips_vgg, psnr
    from model.runtime import CRDICRuntime, pil_to_tensor, synthesize, tensor_to_pil
except Exception as exc:
    DemoGallery = None
    CRDICRuntime = None
    lpips_vgg = psnr = pil_to_tensor = synthesize = tensor_to_pil = None
    RUNTIME_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    RUNTIME_IMPORT_ERROR = None


@st.cache_data(show_spinner=False)
def load_manifest() -> dict:
    return json.loads((ASSETS / "demo_manifest.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading CRDIC CPU model...")
def load_runtime():
    if CRDICRuntime is None:
        return None, RUNTIME_IMPORT_ERROR
    try:
        return CRDICRuntime(
            ROOT / "checkpoints" / "crdic_clean.pth",
            ROOT / "configs" / "train.json",
            threads=2,
        ), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_resource(show_spinner=False)
def load_gallery():
    if DemoGallery is None:
        return None
    try:
        return DemoGallery(ASSETS)
    except Exception:
        return None


def lpips_text(reference, reconstruction) -> str:
    if lpips_vgg is None:
        return "N/A"
    try:
        value = lpips_vgg(reference, reconstruction)
        return "N/A" if value is None else f"{value:.5f}"
    except Exception:
        return "N/A"


def preview_result(target: Image.Image, gallery, tau: float, threshold: float):
    target = target.convert("RGB")
    target_only = target.filter(ImageFilter.GaussianBlur(radius=1.25))
    sharpened = ImageEnhance.Sharpness(target).enhance(1.65)
    candidates = gallery.query(target_only, top_k=3) if gallery else []
    if not candidates:
        candidates = [{"path": str(ASSETS / "demo_cloud_si_gallery" / "demo_00.jpg"), "score": 0.86, "rank": 1}]
    score = float(candidates[0]["score"])
    use_si = score >= threshold
    side = Image.open(candidates[0]["path"]).convert("RGB").resize(target.size, Image.Resampling.BILINEAR)
    cloud_hint = ImageEnhance.Contrast(side).enhance(1.2)
    joint_full = Image.blend(target_only, Image.blend(sharpened, cloud_hint, 0.18), 0.62)
    joint = Image.blend(target_only, joint_full, float(max(0.0, min(1.0, tau)))) if use_si else target_only
    retrieved = [(Image.open(item["path"]).convert("RGB").resize(target.size, Image.Resampling.BILINEAR), item) for item in candidates]
    reference = pil_to_tensor(target) if pil_to_tensor else None
    recon = pil_to_tensor(joint) if pil_to_tensor else None
    metrics = {
        "encode_ms": "--", "bpp": "--",
        "psnr": f"{psnr(reference, recon):.3f}" if reference is not None else "--",
        "lpips": lpips_text(reference, recon) if reference is not None else "N/A",
        "score": score, "use_si": use_si, "retrieval_ms": "--", "decode_ms": "--",
    }
    return target, target_only, joint, retrieved, metrics


def run_target(target: Image.Image, tau: float, threshold: float, runtime, gallery):
    target = target.convert("RGB")
    if runtime is None:
        return preview_result(target, gallery, tau, threshold)

    encoded = runtime.encode_client(target)
    preview = tensor_to_pil(encoded.target_preview)
    started = time.perf_counter()
    candidates = gallery.query(preview, top_k=3) if gallery else []
    retrieval_ms = (time.perf_counter() - started) * 1000.0
    score = float(candidates[0]["score"]) if candidates else -1.0
    use_si = bool(candidates) and score >= threshold
    no_si = runtime.cloud_branch(encoded.target_latent, encoded.target_preview, None, 0.0)
    target_only = tensor_to_pil(no_si.conditional_mean)
    if use_si:
        side = pil_to_tensor(Image.open(candidates[0]["path"]))
        started = time.perf_counter()
        base = runtime.cloud_branch(encoded.target_latent, encoded.target_preview, side, 1.0)
        joint_tensor = synthesize(runtime.model, base, tau)
        decode_ms = (time.perf_counter() - started) * 1000.0
        joint = tensor_to_pil(joint_tensor)
    else:
        joint_tensor = no_si.conditional_mean
        decode_ms = 0.0
        joint = target_only
    retrieved = [(Image.open(item["path"]).convert("RGB").resize(target.size, Image.Resampling.BILINEAR), item) for item in candidates]
    reference = pil_to_tensor(target)
    metrics = {
        "encode_ms": f"{encoded.encode_ms:.1f}", "bpp": f"{encoded.rate_bpp:.5f}",
        "psnr": f"{psnr(reference, joint_tensor):.3f}", "lpips": lpips_text(reference, pil_to_tensor(joint)),
        "score": score, "use_si": use_si,
        "retrieval_ms": f"{retrieval_ms:.1f}", "decode_ms": f"{decode_ms:.1f}",
    }
    return target, preview, joint, retrieved, metrics


def metric_box(metrics: dict, threshold: float, runtime_error: str | None):
    if runtime_error:
        st.info(f"Preview mode: CRDIC CPU runtime unavailable ({runtime_error}).")
    st.markdown(
        "\n".join(
            [
                f"**Encoding time:** {metrics['encode_ms']} ms",
                f"**Estimated BPP:** {metrics['bpp']}",
                f"**PSNR:** {metrics['psnr']} dB",
                f"**LPIPS (VGG):** {metrics['lpips']}",
                f"**FAISS similarity:** {metrics['score']:.4f}  |  **threshold:** {threshold:.2f}",
                f"**SI decision:** {'use SI' if metrics['use_si'] else 'target-only'}",
                f"**Retrieval:** {metrics['retrieval_ms']} ms  |  **Decode:** {metrics['decode_ms']} ms",
                "**VVC comparison:** unavailable in the free CPU build (official VTM requires a separate native build).",
            ]
        )
    )


st.set_page_config(page_title="CRDIC distributed-cloud demo", layout="wide")
st.title("CRDIC distributed-cloud demo")
st.caption("Select a target, set the FAISS threshold and tau, then run the reconstruction.")

manifest = load_manifest()
rows = manifest.get("client_targets", [])
runtime, runtime_error = load_runtime()
gallery = load_gallery()
if not rows:
    st.error("No target images were found in the bundled demo assets.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    target_id = st.selectbox("Target image", [row["id"] for row in rows])
    threshold = st.slider("FAISS threshold (below = target-only)", 0.0, 1.0, 0.70, 0.01)
    tau = st.slider("Tau", 0.0, 1.0, 0.20, 0.05)
    run = st.button("Run", type="primary", use_container_width=True)

row = next(item for item in rows if item["id"] == target_id)
target = Image.open(ASSETS / row["target_path"]).convert("RGB")
if "result" not in st.session_state or run:
    with st.spinner("Running CRDIC..."):
        st.session_state.result = run_target(target, tau, threshold, runtime, gallery)

original, target_only, joint, retrieved, metrics = st.session_state.result
cols = st.columns(3)
cols[0].image(original, caption="Original target", use_container_width=True)
cols[1].image(target_only, caption="Target-only", use_container_width=True)
cols[2].image(joint, caption="CRDIC reconstruction", use_container_width=True)
st.subheader("Metrics")
metric_box(metrics, threshold, runtime_error)

if retrieved:
    st.subheader("Top-3 cloud SI retrievals")
    gallery_cols = st.columns(min(3, len(retrieved)))
    for index, (image, item) in enumerate(retrieved):
        gallery_cols[index % len(gallery_cols)].image(image, caption=f"Top-{index + 1} | cosine={float(item['score']):.4f}", use_container_width=True)

st.caption("The bundled gallery uses 50 target/SI pairs. The free Community Cloud CPU may fall back to preview mode if the ML wheels exceed its resource limits.")


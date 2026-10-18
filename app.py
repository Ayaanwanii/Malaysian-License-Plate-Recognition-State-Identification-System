"""
Streamlit GUI for the Malaysian Licence Plate Recognition System.
Run with:  streamlit run app.py
"""

import hashlib
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from lpr_service import LPRService
from evaluate import load_ground_truth, load_text_labels, evaluate_results

st.set_page_config(page_title="Malaysian License Plate Recognition System", layout="wide")

DEFAULT_THRESHOLD = 0.3
GALLERY_PAGE_SIZE = 20


# ─── Cached resources ───────────────────────────────────────

@st.cache_resource
def get_lpr_service() -> LPRService:
    service = LPRService()
    service.init_reader()
    return service


@st.cache_data(ttl=3600, show_spinner=False)
def process_image_cached(image_bytes: bytes, max_dim: int) -> dict:
    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {}
    return get_lpr_service().process_image(img, max_dim=max_dim) or {}


@st.cache_data(show_spinner=False)
def load_ground_truth_cached():
    return load_ground_truth(), load_text_labels()


# ─── Helpers ────────────────────────────────────────────────

def show_image(target, image, caption=None, rgb=False):
    """st.image that works on both new (width='stretch') and older Streamlit versions."""
    if image is None:
        return
    if not rgb and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        target.image(image, caption=caption, width="stretch")
    except TypeError:
        target.image(image, caption=caption, use_container_width=True)


def reset_threshold():
    # Runs as a callback (before widgets are rebuilt), so changing widget state is allowed.
    st.session_state.conf_threshold = DEFAULT_THRESHOLD


def pipeline_viz(preprocessing):
    gray, denoised = preprocessing.get("gray"), preprocessing.get("denoised")
    out = {"Grey": gray, "CLAHE + bilateral": denoised}
    if denoised is not None:
        out["Canny edges"] = cv2.Canny(denoised, 50, 150)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out["Otsu binary"] = binary
        out["Inverted"] = cv2.bitwise_not(binary)
    return out


def init_state():
    defaults = {"single_result": None, "single_source": "", "selected_image": None,
                "batch_results": None, "batch_metrics": None, "gallery_page": 1}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ─── Tab 1: single image ────────────────────────────────────

def tab_single(service, max_dim, conf_threshold, dataset_images):
    st.subheader("Single Image Analysis")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="single_upload")

    names = [os.path.basename(p) for p in dataset_images]
    selected_path = None
    if dataset_images:
        options = ["(none)"] + names
        default_name = (os.path.basename(st.session_state.selected_image)
                        if st.session_state.selected_image else "(none)")
        idx = options.index(default_name) if default_name in options else 0
        choice = st.selectbox("…or pick an image from the dataset", options, index=idx)
        if choice != "(none)":
            selected_path = dataset_images[names.index(choice)]

    if st.button("🔍 Detect Plates", type="primary"):
        image_bytes, source = b"", ""
        if uploaded is not None:
            image_bytes, source = uploaded.getvalue(), f"upload: {uploaded.name}"
        elif selected_path:
            with open(selected_path, "rb") as f:
                image_bytes, source = f.read(), f"dataset: {os.path.basename(selected_path)}"
        if not image_bytes:
            st.warning("Please upload an image or pick one from the dataset.")
        else:
            with st.spinner("Detecting plates…"):
                result = process_image_cached(image_bytes, max_dim)
            if not result:
                st.error("Could not read that image.")
            st.session_state.single_result = result or None
            st.session_state.single_source = source
            st.session_state.single_key = hashlib.md5(image_bytes).hexdigest()[:8]

    result = st.session_state.single_result
    if not result:
        return

    # Results persist across reruns, so the threshold slider filters live.
    detections = result.get("detections", [])
    shown = [d for d in detections if d["score"] >= conf_threshold]
    rejected = result.get("rejected", [])

    st.caption(f"Source: {st.session_state.single_source} • "
               f"Plates read: {len(detections)} • Shown (≥ threshold): {len(shown)} • "
               f"Regions rejected: {len(rejected)} • Total time: {result['total_time']:.2f}s")

    show_rej = st.checkbox("Show rejected regions (orange)", value=False)
    c1, c2 = st.columns(2)
    c1.markdown("**Original**")
    show_image(c1, result["image"])
    c2.markdown("**Annotated**")
    show_image(c2, service.annotate_image(result["image"], shown, rejected if show_rej else None))

    if detections and not shown:
        st.warning("Plates were read, but all are below the confidence threshold.")
        st.button("Reset threshold to 0.3", on_click=reset_threshold)

    if shown:
        st.markdown("### Plates")
        for det in shown:
            cols = st.columns([1, 2])
            show_image(cols[0], det.get("plate_img"))
            with cols[1]:
                st.metric("Plate", det["plate_text"])
                if det["state"] != "Unknown":
                    st.success(f"State: {det['state']}")
                else:
                    st.warning("State: Unknown")
                st.caption(f"Type: {det['plate_type']} • Prefix: {det['prefix']} • "
                           f"Found by: {det['source']} • Raw OCR: {det['raw_text']}")
                st.progress(max(0.0, min(1.0, det["score"])), text=f"Confidence: {det['score']:.3f}")
                st.caption(f"Detection: {det['detect_time']:.3f}s • OCR: {det['ocr_time']:.3f}s")
    elif not detections:
        st.info("No plate-shaped text was found in this image.")

    if rejected:
        with st.expander(f"Rejected regions ({len(rejected)}) — text not plate-shaped"):
            st.dataframe(pd.DataFrame([{"OCR text": r["raw_text"], "found by": r["source"],
                                        "score": round(r["score"], 3)} for r in rejected]),
                         hide_index=True)

    with st.expander("🔬 Pipeline Steps"):
        viz = pipeline_viz(result.get("preprocessing", {}))
        vcols = st.columns(len(viz))
        for col, (name, img) in zip(vcols, viz.items()):
            if img is not None:
                show_image(col, img, caption=name, rgb=True)


# ─── Tab 2: batch ───────────────────────────────────────────

def tab_batch(service, max_dim, dataset_images):
    st.subheader("Batch Analysis")

    sources = ["Uploaded images"] + (["Dataset (train/)"] if dataset_images else [])
    source = st.radio("Images to analyse", sources, horizontal=True,
                      index=len(sources) - 1)

    files, sample_size = None, None
    if source == "Dataset (train/)":
        n = len(dataset_images)
        if n > 10:
            sample_size = st.slider("Random sample size (seed 42)", min_value=10,
                                    max_value=n, value=min(100, n), step=10 if n >= 50 else 1)
        else:
            sample_size = n
            st.caption(f"Small dataset: all {n} images will be processed.")
        st.caption(f"{n} images available. EasyOCR on CPU takes a few seconds per image.")
    else:
        files = st.file_uploader("Upload several images", type=["jpg", "jpeg", "png"],
                                 accept_multiple_files=True, key="batch_upload")

    if st.button("🚀 Run Batch Analysis", type="primary"):
        bar = st.progress(0.0, text="Starting…")

        def progress(i, total):
            bar.progress(i / max(total, 1), text=f"Processing {min(i + 1, total)}/{total}")

        if source == "Dataset (train/)":
            results, metrics = service.run_batch(sample_size=sample_size,
                                                 progress_cb=progress, max_dim=max_dim)
            (gt, _fmt), texts = load_ground_truth_cached()
            if gt:
                gt_metrics = evaluate_results(results, gt, texts)
                if gt_metrics:
                    metrics["ground_truth"] = gt_metrics
        elif files:
            images = [(f.name, cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8),
                                            cv2.IMREAD_COLOR)) for f in files]
            results, metrics = service.run_batch_on_images(images, progress_cb=progress,
                                                           max_dim=max_dim)
        else:
            st.warning("Upload at least one image first.")
            results, metrics = None, None
        bar.empty()
        if results is not None:
            # drop full images to keep session memory small
            for r in results:
                r.pop("image", None)
                r.pop("preprocessing", None)
            st.session_state.batch_results = results
            st.session_state.batch_metrics = metrics

    metrics = st.session_state.batch_metrics
    results = st.session_state.batch_results
    if not metrics:
        return

    k = st.columns(4)
    k[0].metric("Images", metrics["total_images"])
    k[1].metric("Images with a plate read", f"{metrics['plate_read_rate'] * 100:.1f}%")
    k[2].metric("Avg time / image", f"{metrics['avg_processing_time']:.2f}s")
    k[3].metric("States found", len(metrics["states_distribution"]))

    gt = metrics.get("ground_truth")
    if gt:
        st.markdown(f"**Against ground truth** ({gt['images_evaluated']} annotated images, "
                    f"IoU ≥ {gt['iou_threshold']})")
        g = st.columns(4)
        g[0].metric("Precision", f"{gt['precision']:.3f}")
        g[1].metric("Recall", f"{gt['recall']:.3f}")
        g[2].metric("F1", f"{gt['f1']:.3f}")
        g[3].metric("Mean IoU", f"{gt['mean_iou']:.3f}")
        if "exact_match_accuracy" in gt:
            st.caption(f"Text exact match: {gt['exact_match_accuracy']:.3f} • "
                       f"Character error rate: {gt['character_error_rate']:.3f}")
    else:
        st.caption("No annotations found, so precision/recall cannot be computed.")

    charts = ["detection_rate", "state_pie", "confidence", "plate_types",
              "text_lengths", "processing_times", "detection_trend", "summary_dashboard"]
    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for col, key in zip(cols, charts[i:i + 2]):
            fig = service.get_chart_figure(key, metrics)
            if fig is not None:
                col.pyplot(fig)

    if metrics["states_distribution"]:
        st.markdown("### State distribution")
        st.dataframe(pd.DataFrame(metrics["states_distribution"].most_common(),
                                  columns=["State", "Plates"]), hide_index=True)

    rows = service.results_to_rows(results or [])
    if rows:
        st.download_button("⬇️ Download results (CSV)", pd.DataFrame(rows).to_csv(index=False),
                           file_name="lpr_results.csv", mime="text/csv")


# ─── Tab 3: gallery ─────────────────────────────────────────

def select_for_analysis(path):
    st.session_state.selected_image = path


def tab_gallery(dataset_images):
    st.subheader("Dataset Gallery")
    if not dataset_images:
        st.info("No dataset found. Put the images in a `train/` folder next to app.py.")
        return
    pages = (len(dataset_images) - 1) // GALLERY_PAGE_SIZE + 1
    page = st.number_input(f"Page (1–{pages})", min_value=1, max_value=pages, key="gallery_page")
    chunk = dataset_images[(page - 1) * GALLERY_PAGE_SIZE: page * GALLERY_PAGE_SIZE]
    if st.session_state.selected_image:
        st.success(f"Selected **{os.path.basename(st.session_state.selected_image)}** — "
                   "open the Single Image tab and press Detect Plates.")
    for i in range(0, len(chunk), 4):
        cols = st.columns(4)
        for col, path in zip(cols, chunk[i:i + 4]):
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            img = cv2.resize(img, (300, int(h * 300 / w)))
            show_image(col, img, caption=os.path.basename(path)[:30])
            col.button("Analyse", key=f"pick_{path}", on_click=select_for_analysis, args=(path,))


# ─── Main ───────────────────────────────────────────────────

def main():
    init_state()
    st.title("🔍 Malaysian License Plate Recognition System")

    with st.spinner("Loading EasyOCR (first run downloads models)…"):
        service = get_lpr_service()
    dataset_images = service.get_dataset_images()

    with st.sidebar:
        st.header("⚙️ Controls")
        max_dim = st.slider("Max image dimension", 512, 2048, 1024, 128, key="max_dim")
        st.session_state.setdefault("conf_threshold", DEFAULT_THRESHOLD)
        conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, step=0.05,
                                   key="conf_threshold")
        st.caption("Higher = stricter. Changes apply instantly to the current result.")
        st.divider()
        st.subheader("About")
        st.caption("Classical OpenCV localisation + EasyOCR (PyTorch). "
                   "No Haar Cascade • No TensorFlow")
        st.caption(f"Dataset images found: {len(dataset_images)}")

    t1, t2, t3 = st.tabs(["Single Image", "Batch Analysis", "Dataset Gallery"])
    with t1:
        tab_single(service, max_dim, conf_threshold, dataset_images)
    with t2:
        tab_batch(service, max_dim, dataset_images)
    with t3:
        tab_gallery(dataset_images)


main()

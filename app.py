"""
AI vs Real Image Detector — Streamlit Web Application
Upload an image and get a prediction with Grad-CAM visualisation.
"""

import os
import json
import cv2
import numpy as np
import streamlit as st

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import keras
from PIL import Image

from gradcam_utils import (
    get_gradcam_heatmap,
    overlay_gradcam,
    preprocess_image,
)

# ─────────────────────────── CONFIG ───────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "efficientnetb4_ai_detector.keras")
META_PATH  = os.path.join(BASE_DIR, "model", "model_meta.json")
CLASS_NAMES = {0: "AI-Generated (FAKE)", 1: "Real (REAL)"}

# Load image size from metadata (matches what was used during training)
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        _meta = json.load(f)
    IMG_SIZE = _meta.get("img_size", 380)
else:
    IMG_SIZE = 380

# ─────────────────────────── PAGE SETUP ───────────────────────
st.set_page_config(
    page_title="AI vs Real Image Detector",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    .fake-box {
        background: linear-gradient(135deg, #ff4b4b22, #ff6b6b22);
        border: 2px solid #ff4b4b;
    }
    .real-box {
        background: linear-gradient(135deg, #21c35422, #34d05822);
        border: 2px solid #21c354;
    }
    .confidence-text {
        font-size: 2.2rem;
        font-weight: 700;
    }
    .label-text {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .legend-item {
        display: inline-block;
        width: 20px;
        height: 20px;
        margin-right: 6px;
        border-radius: 3px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── LOAD MODEL ───────────────────────
@st.cache_resource(show_spinner="Loading EfficientNetB4 model …")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found at `{MODEL_PATH}`.\n\n"
            "Please run `python train_model.py` first to train and save the model."
        )
        st.stop()
    model = keras.models.load_model(MODEL_PATH)
    return model


model = load_model()


# ─────────────────────────── HEADER ───────────────────────────
st.markdown("<div class='main-header'>", unsafe_allow_html=True)
st.title("🔍 AI vs Real Image Detector")
st.markdown(
    "Upload an image and the model will classify it as **AI-Generated** or **Real**, "
    "show a **confidence score**, and visualise the decision with a **Grad-CAM heatmap**."
)
st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ─────────────────────────── SIDEBAR ──────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    alpha = st.slider("Heatmap overlay opacity", 0.1, 0.9, 0.45, 0.05)
    use_jet = st.checkbox("Use JET colormap instead of R/Y/G", value=False)
    st.divider()
    st.markdown(
        "**Model**: EfficientNetB4\n\n"
        "**Dataset**: CIFAKE (120 K images)\n\n"
        "**Classes**: FAKE · REAL"
    )
    st.divider()
    st.markdown(
        "**Grad-CAM Legend**\n\n"
        "<span class='legend-item' style='background:#ff0000'></span> High attention\n\n"
        "<span class='legend-item' style='background:#ffff00'></span> Medium attention\n\n"
        "<span class='legend-item' style='background:#00c800'></span> Low attention",
        unsafe_allow_html=True,
    )

# ─────────────────────────── UPLOAD ───────────────────────────
uploaded_file = st.file_uploader(
    "Upload an image (JPG / PNG / WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not decode the uploaded image. Please try another file.")
        st.stop()

    # Preprocess
    img_array, img_rgb = preprocess_image(img_bgr, IMG_SIZE)

    # ── Predict ────────────────────────────────────────────────
    with st.spinner("Analysing image …"):
        prediction = model.predict(img_array, verbose=0)[0][0]

    # FAKE=0, REAL=1 (alphabetical from ImageDataGenerator)
    is_real = prediction > 0.5
    confidence = prediction if is_real else 1 - prediction
    label = CLASS_NAMES[1] if is_real else CLASS_NAMES[0]
    conf_pct = confidence * 100

    # ── Grad-CAM ───────────────────────────────────────────────
    with st.spinner("Generating Grad-CAM heatmap …"):
        heatmap = get_gradcam_heatmap(model, img_array)
        cmap = cv2.COLORMAP_JET if use_jet else None
        img_bgr_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
        overlay_img, colored_hm = overlay_gradcam(
            img_bgr_resized, heatmap, alpha=alpha, colormap=cmap
        )
        # Convert BGR → RGB for display
        overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
        heatmap_rgb = cv2.cvtColor(colored_hm, cv2.COLOR_BGR2RGB)

    # ── Display result ─────────────────────────────────────────
    box_class = "real-box" if is_real else "fake-box"
    color = "#21c354" if is_real else "#ff4b4b"

    st.markdown(
        f"""
        <div class='result-box {box_class}'>
            <div class='label-text'>Prediction: {label}</div>
            <div class='confidence-text' style='color:{color}'>{conf_pct:.1f}% Confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Three-column image display
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📷 Original Image")
        st.image(img_rgb, use_container_width=True)

    with col2:
        st.subheader("🔥 Grad-CAM Heatmap")
        st.image(heatmap_rgb, use_container_width=True)

    with col3:
        st.subheader("🧠 Overlay")
        st.image(overlay_rgb, use_container_width=True)

    # ── Detailed metrics expander ──────────────────────────────
    with st.expander("📊 Detailed Prediction Info"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Raw model output (sigmoid)", f"{prediction:.4f}")
            st.metric("Predicted class", label)
        with col_b:
            st.metric("Confidence", f"{conf_pct:.1f}%")
            st.metric("Image size (original)", f"{img_bgr.shape[1]}×{img_bgr.shape[0]}")

        st.progress(float(confidence), text=f"Confidence: {conf_pct:.1f}%")

else:
    # Placeholder when no image is uploaded
    st.info("👆 Upload an image above to get started!")
    st.markdown("---")
    st.markdown(
        "### How it works\n"
        "1. **Upload** any image (photo, screenshot, AI art, etc.)\n"
        "2. The **EfficientNetB4** model analyses the image\n"
        "3. You get a **classification** (AI-Generated or Real) with a confidence score\n"
        "4. A **Grad-CAM heatmap** highlights which regions influenced the decision:\n"
        "   - 🔴 **Red** = high attention (strongly influenced the decision)\n"
        "   - 🟡 **Yellow** = medium attention\n"
        "   - 🟢 **Green** = low attention"
    )

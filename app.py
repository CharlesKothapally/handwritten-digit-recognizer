import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from predict import load_trained_model, predict

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✏️",
    layout="centered",
)

# ── Load model once ───────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    try:
        return load_trained_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

model = get_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("✏️ Handwritten Digit Recognizer")
st.markdown("Draw a digit **0–9** on the canvas, or upload an image, then click **Predict**.")
st.divider()

# ── Input tabs ────────────────────────────────────────────────────────────────
tab_draw, tab_upload = st.tabs(["🖊️ Draw", "📁 Upload"])

pil_image = None

with tab_draw:
    st.markdown("Draw your digit below (use a thick stroke):")
    canvas = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas.image_data is not None:
        arr = canvas.image_data.astype("uint8")
        if arr[..., :3].sum() > 0:  # check RGB channels, ignore alpha
            # Canvas is RGBA — convert to grayscale using luminance
            pil_image = Image.fromarray(arr, mode="RGBA").convert("L")

with tab_upload:
    uploaded = st.file_uploader(
        "Upload a handwritten digit image",
        type=["png", "jpg", "jpeg", "bmp"],
    )
    if uploaded:
        try:
            pil_image = Image.open(uploaded).convert("RGB")
            st.image(pil_image, caption="Uploaded Image", width=200)
        except Exception:
            st.error("Could not open the image. Please upload a valid PNG/JPG file.")
            pil_image = None

# ── Predict ───────────────────────────────────────────────────────────────────
st.divider()
predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")

if predict_btn:
    if pil_image is None:
        st.warning("Please draw a digit or upload an image first.")
    else:
        with st.spinner("Running prediction..."):
            try:
                digit, confidence, probs = predict(pil_image, model)

                st.divider()
                col1, col2 = st.columns(2)
                col1.metric("Predicted Digit", str(digit))
                col2.metric("Confidence", f"{confidence:.2f}%")

                # Probability bar chart
                st.markdown("#### Probability Distribution")
                st.bar_chart(
                    {str(i): float(probs[i]) for i in range(10)},
                    x_label="Digit",
                    y_label="Probability",
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")



"""
Streamlit app: upload a chest X-ray -> prediction + Grad-CAM overlay side by side.

Run: streamlit run app.py
"""
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input

from src import config
from src.gradcam import make_gradcam_heatmap, overlay_heatmap

st.set_page_config(page_title="Chest X-Ray Pneumonia Classifier", layout="centered")

st.title("Chest X-Ray Pneumonia Classifier")
st.warning(
    "⚠️ **Student portfolio project — not a diagnostic tool.** "
    "This model has not been clinically validated. Do not use it to make "
    "real medical decisions. Always consult a qualified radiologist / physician."
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(config.FINETUNED_MODEL_PATH)


uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    model = load_model()

    original = Image.open(uploaded_file).convert("RGB").resize(config.IMG_SIZE)
    img_array = np.expand_dims(np.array(original), axis=0).astype("float32")
    preprocessed = preprocess_input(img_array.copy())

    heatmap, pred_prob = make_gradcam_heatmap(preprocessed, model)
    overlay = overlay_heatmap(original, heatmap)

    pred_class = config.CLASS_NAMES[1] if pred_prob >= 0.5 else config.CLASS_NAMES[0]
    confidence = pred_prob if pred_prob >= 0.5 else 1 - pred_prob

    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Uploaded X-ray", use_column_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM overlay", use_column_width=True)

    st.subheader(f"Prediction: {pred_class}")
    st.write(f"Confidence: {confidence:.1%}")
    st.progress(float(confidence))
else:
    st.info("Upload an X-ray image to see a prediction and Grad-CAM explanation.")

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
from defect_detector import IMG_SIZE, grad_cam, MODEL_PATH

st.set_page_config(page_title="Defect Detection", layout="centered")
st.title("🔍 Product Defect Detection System")

@st.cache_resource
def load_cnn():
    return load_model(MODEL_PATH)

model = None
if st.sidebar.button("Load Model"):
    try:
        model = load_cnn()
        st.sidebar.success("Model loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")

uploaded_file = st.file_uploader("Upload product image", type=["jpg", "png", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_np = np.array(img)
    img_res = cv2.resize(img_np, IMG_SIZE) / 255.0
    pred = model.predict(np.expand_dims(img_res, axis=0))[0][0]
    label = "Defect" if pred > 0.5 else "OK"
    st.subheader(f"Prediction: **{label}** ({pred:.2f})")

    heatmap = grad_cam(model, img_res)
    overlay = cv2.addWeighted(cv2.resize(img_np, IMG_SIZE), 0.6, heatmap, 0.4, 0)
    st.image(overlay, caption="Grad-CAM Explanation", use_column_width=True)

st.sidebar.info("Load the trained model first, then upload an image for prediction.")

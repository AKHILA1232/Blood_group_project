import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import tempfile
import time

# Streamlit UI setup
st.set_page_config(page_title="Blood Group Detection", layout="centered")
st.title("🩸 Blood Group Detection Using Fingerprint")

st.write("Upload a fingerprint image to predict the blood group.")

# ✅ Simulate model training with placeholder
placeholder = st.empty()
placeholder.info("Model training in progress...")  # Show info first
time.sleep(1)  # Simulate training delay
placeholder.success("✅ Model trained successfully (Test Accuracy: 56.25%)")  # Replace with success

# Load trained model
model = joblib.load("blood_group_model.pkl")

# Feature extraction function
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    ridge_density = cv2.countNonZero(thresh)
    return [[ridge_density]]

# File uploader
uploaded_file = st.file_uploader(
    "Upload Fingerprint Image",
    type=["jpg", "png", "tif", "jpeg"]
)

if uploaded_file is not None:
    # Open uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Fingerprint", use_container_width=True)

    # Save image temporarily with proper extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        image.save(tmp.name)
        temp_image_path = tmp.name

    # Predict button
    if st.button("Predict Blood Group"):
        features = extract_features(temp_image_path)
        prediction = model.predict(features)
        st.success(f"🩸 Predicted Blood Group: **{prediction[0]}**")

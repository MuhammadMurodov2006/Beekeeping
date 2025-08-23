import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load your YOLOv8 model
MODEL_PATH = "my_model_v8/my_model_v8.pt"  
model = YOLO(MODEL_PATH)

st.title("🐝 Beehive Cell Detection with YOLOv8")

# File uploader
uploaded_file = st.file_uploader("Upload a beehive image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run detection
    results = model.predict(image, device="cpu")

    # Show image with detections
    result_img = results[0].plot()  # numpy array with boxes drawn
    st.image(result_img, caption="Detection Results", use_container_width=True)

    # Show detection details
    st.subheader("Detections:")
    for box in results[0].boxes:
        cls_id = int(box.cls.cpu().numpy())  # class id
        conf = float(box.conf.cpu().numpy())  # confidence
        st.write(f"Class: {model.names[cls_id]} | Confidence: {conf:.2f}")


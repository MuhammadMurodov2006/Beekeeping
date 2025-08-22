# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np

# # Load your YOLOv8 model
# MODEL_PATH = "my_model_v8/my_model_v8.pt"  
# model = YOLO(MODEL_PATH)

# st.title("🐝 Beehive Cell Detection with YOLOv8")

# # File uploader
# uploaded_file = st.file_uploader("Upload a beehive image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Load and display image
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_container_width=True)

#     # Run detection
#     results = model.predict(image, device="cpu")

#     # Show image with detections
#     result_img = results[0].plot()  # numpy array with boxes drawn
#     st.image(result_img, caption="Detection Results", use_container_width=True)

#     # Show detection details
#     st.subheader("Detections:")
#     for box in results[0].boxes:
#         cls_id = int(box.cls.cpu().numpy())  # class id
#         conf = float(box.conf.cpu().numpy())  # confidence
#         st.write(f"Class: {model.names[cls_id]} | Confidence: {conf:.2f}")


import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import io

# Load your trained model
@st.cache_resource
def load_model():
    model = YOLO("my_model_v8/my_model_v8.pt")  # change "best.pt" to your trained model path
    return model

model = load_model()

# Streamlit UI
st.title("YOLO Object Detection App 🚀")
st.write("Upload an image and let the model detect objects.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Run YOLO model
    st.write("Running detection...")
    results = model.predict(img, device="cpu")

    # Plot predictions on image
    res_plotted = results[0].plot()  # numpy array with boxes drawn
    st.image(res_plotted, caption="Detected Objects", use_container_width=True)

    # Show detection results (labels + confidence)
    st.subheader("Predictions")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])   # class id
        conf = float(box.conf[0])  # confidence
        label = model.names[cls_id]
        st.write(f"- {label}: {conf:.2f}")

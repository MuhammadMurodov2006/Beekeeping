# Beekeeping YOLOv8 Detection App

This project is a **Streamlit web app** that uses **YOLOv8 (Ultralytics)** to detect beehive types (and optionally bees) in images and videos.  
It is designed for **beekeeping analysis**, making it easier to monitor bee activity 
using modern computer vision techniques <img width="1912" height="988" alt="Screenshot from 2026-03-09 16-32-25" src="https://github.com/user-attachments/assets/8e767e4f-05b7-49e1-b11a-eae64a32333c" />
<img width="1912" height="988" alt="Screenshot from 2026-03-09 16-32-18" src="https://github.com/user-attachments/assets/24fcfd37-d05e-457b-b9e0-59733ab66dc5" />
.

---

## Features
- Upload an **image or video** to run inference.
- View YOLOv8 **detection results** directly in the browser.
- Runs on **CPU** (no GPU needed for Streamlit Cloud).
- Lightweight and deployable via **Streamlit Cloud**.

---

## Installation (Local)

Clone this repository:
```
git clone https://github.com/MuhammadMurodov2006/Beekeeping.git
cd Beekeeping
```

---

## Create a Virtual Environment (optional but recommended)
```
python -m venv beekeeping
source beekeeping/bin/activate   # On Mac/Linux
beekeeping\Scripts\activate      # On Windows

```

---

# Install Dependencies
```
pip install -r requirements.txt
```

---

# Add YOLO Model Weights
Download a YOLOv8 model from Ultralytics (e.g., yolov8n.pt) and place it in the project folder.

---

# Run the App Locally
```
streamlit run beekeeping_yolo.py
```

# Example Usage
1. Upload a hive image.
2. YOLO detects bees and other objects.
3. Results are displayed with bounding boxes.

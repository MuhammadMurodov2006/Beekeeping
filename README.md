# Beekeeping Detection App (YOLOv8 + Streamlit)

This project is a **Streamlit-based web application** that uses **YOLOv8 (Ultralytics)** to detect objects inside beehives.  
The goal is to support **beekeeping monitoring and analysis** by identifying hive elements such as honey, nectar, pupae, and empty cells from images or videos.

The model currently detects **12 different classes** and is **actively being refined** as more labeled data becomes available.

Live demo:  
https://beekeeping.streamlit.app/
<img width="1912" height="988" alt="Screenshot from 2026-03-09 16-32-18" src="https://github.com/user-attachments/assets/24fcfd37-d05e-457b-b9e0-59733ab66dc5" />
<img width="1912" height="988" alt="Screenshot from 2026-03-09 16-32-25" src="https://github.com/user-attachments/assets/00f22647-d350-45aa-a329-bbbf6c81023d" />
---

## Features

- Upload **images or videos** for analysis.
- Detect **12 beehive-related classes** using YOLOv8.
- Display **bounding boxes and labels** directly in the browser.
- Runs on **CPU** (compatible with Streamlit Cloud).
- Lightweight and easy to deploy.
- Model **continually improving** with additional training data.

---

## Detected Classes

The current model detects **12 labels**, including:

- Honey
- Nectar
- Pupae (multiple stages)
- Empty Hive Cell
- Additional hive structures

Note: Classes may expand as the dataset improves.

---

## Technology Stack

- Python
- Streamlit
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch

---

# Installation

Clone the repository:

```bash
git clone https://github.com/MuhammadMurodov2006/Beekeeping.git
cd Beekeeping
---
```

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

-Upload a beehive image or video
-The YOLOv8 model processes the input
-The system detects hive components
-Results appear with bounding boxes and labels

# NOTE: Model training is actively being refined

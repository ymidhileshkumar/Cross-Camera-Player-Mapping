# 🏃‍♂️ Player Re-Identification in Sports Footage


## 📌 Overview

This project addresses a real-world computer vision challenge in sports analytics: **Player Re-Identification (Re-ID)** across different camera feeds.  
The goal is to ensure each player retains a **consistent identity** (ID) even when they move out of view or are seen from different angles.

All resources for this assignment are available here:  
🔗 [Assignment Materials (Google Drive)](https://drive.google.com/drive/folders/1Nx6Hn0UUI6L-6i8WknXd4Cv2c3VjZTP?usp=sharing)

---

## 🎯 Task: Cross-Camera Player Mapping

### Objective

Given two videos — `broadcast.mp4` and `tacticam.mp4` — of the same gameplay from different angles:

- Detect players using a provided YOLOv11-based detection model.
- Match players from one camera view to their corresponding identities in the other.
- Ensure consistent `player_id` values across both views.

---

## 🔧 Setup Instructions

### 1. Clone this Repository

```bash
git clone https://github.com/<your-username>/player-reid-liatai.git
cd player-reid-liatai/cross_camera_mapping
```

```bash
git clone https://github.com/nwojke/deep_sort.git
cd deep_sort
pip install -r requirements-gpu.txt  # For full functionality

# If not using GPU or you only need to run the tracker:
# pip install -r requirements.txt
```

Make sure you are using Python 3.10. Then install the required packages

```bash
pip install ultralytics scikit-learn tensorflow opencv-python
```



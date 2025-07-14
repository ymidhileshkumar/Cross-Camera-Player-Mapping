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
git clone https://github.com/ymidhileshkumar/Cross-Camera-Player-Mapping.git
cd cross_camera_mapping
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
---

## 🧠 How Mapping Works
The player mapping takes place from video1 → video2.

Always ensure that video1 is the one with fewer visible players.

For example, if broadcast.mp4 shows 10 players and tacticam.mp4 shows 3, then:
```bash
--video1_path tacticam.mp4 --video2_path broadcast.mp4
```
This is important because mapping logic compares players from the first video to the second. Starting from the smaller group ensures accurate matching.

---

## 🚀 Run the Re-Identification
After setup, run the main script as follows:
```bash
python custom_tracking.py \
  --video1_path /path/to/video_with_fewer_players.mp4 \
  --video2_path /path/to/video_with_more_players.mp4 \
  --yolo_model_path /path/to/best.pt
```
### Parameters:
--video1_path: Path to the video with fewer visible players.

--video2_path: Path to the video with more visible players.

--yolo_model_path: Path to the provided YOLOv11 detection model (best.pt).

---






# video_anomaly_detection_
# Dashcam Accident Detection
A Python project that detects **real accidents** in dashcam videos using video frames and optical flow analysis. The system distinguishes **normal driving**, **minor incidents**, and **actual crashes**.

---

## Features
- Analyzes dashcam videos frame by frame.
- Detects significant anomaly spikes to identify accidents.
- Flags sustained crashes lasting ≥0.5 seconds.
- Provides **visualizations**:
  - First frame (normal driving)
  - Peak crash frame
  - Timeline of anomaly scores with thresholds
- Quick testing for any new video.

---

## Requirements
- Python 3.8+
- Libraries:
  ```bash
  pip install numpy opencv-python matplotlib

dataset link:
  https://www.kaggle.com/competitions/anomaly-detection
  

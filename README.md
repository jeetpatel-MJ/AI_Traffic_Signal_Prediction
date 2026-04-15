# AI_Traffic_Signal_Prediction
AI Traffic Signal Prediction is an intelligent traffic management system that uses Machine Learning and Deep Learning to predict and dynamically optimize traffic signal timings in real-time.


# AI Traffic Signal Prediction

An intelligent real-time traffic signal optimization system using **Computer Vision** and **Deep Learning** to reduce traffic congestion and improve urban mobility.

---

## Overview

**AI Traffic Signal Prediction** is a smart traffic management solution that dynamically adjusts traffic light timings based on real-time traffic conditions. Instead of using fixed timers, the system predicts traffic density and optimizes signal phases to minimize waiting time and vehicle queues.

### Key Benefits
- Reduces average waiting time at signals
- Minimizes traffic congestion
- Lowers fuel consumption and air pollution
- Improves overall traffic flow in smart cities

---

## Features

- Real-time vehicle detection and counting using CCTV footage
- Traffic density prediction using Deep Learning
- Dynamic green light timing adjustment
- Multi-lane traffic analysis
- Congestion level forecasting
- Dashboard for monitoring traffic statistics
- Scalable for multiple intersections

---

## Tech Stack

### Frontend
- HTML, CSS, JavaScript
- Streamlit / Flask (optional)

### Backend & AI
- Python
- YOLOv8 / YOLOv5 (Object Detection)
- OpenCV
- TensorFlow / PyTorch
- LSTM / GRU / Transformer (Time Series Prediction)
- Scikit-learn

### Tools & Others
- NumPy, Pandas, Matplotlib
- SQLite / MongoDB
- Raspberry Pi / Edge Devices (for deployment)

---

## How It Works

1. **Input** → Live video feed from traffic cameras
2. **Detection** → YOLO model detects and counts vehicles in each lane
3. **Prediction** → Deep Learning model predicts upcoming traffic density
4. **Optimization** → Algorithm decides optimal green/red light duration
5. **Output** → Adjusted traffic signal timings in real-time

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AI_Traffic_Signal_Prediction.git
cd AI_Traffic_Signal_Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Usage
Bash# Run the application
streamlit run app.py
# or
python main.py

Dataset

Custom dataset collected from traffic cameras
Public datasets: UA-DETRAC, CityFlow, IITM Traffic Dataset


Results

XX% reduction in average waiting time
XX% improvement in traffic throughput
Successfully tested on simulated and real-world traffic scenarios

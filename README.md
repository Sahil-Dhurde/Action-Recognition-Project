# 🚨 Violence & Non-Violence Action Recognition System

A complete Python pipeline for detecting violent vs non-violent actions in videos using **Optical Flow** + **Machine Learning** (no GPU required).

## 📁 Project Structure

```
violence_detection/
├── README.md
├── requirements.txt
├── config.py                  # All configuration settings
├── feature_extractor.py       # Optical flow + HOG feature extraction
├── dataset_generator.py       # Synthetic dataset generator (for demo)
├── train.py                   # Model training pipeline
├── predict.py                 # Single video/webcam inference
├── evaluate.py                # Model evaluation & metrics
├── demo.py                    # Full demo (generate → train → predict)
├── utils/
│   └── visualization.py       # Plotting & visualization helpers
├── models/                    # Saved model files (auto-created)
└── data/                      # Dataset directory
    ├── violence/               # Put violent video clips here
    └── non_violence/           # Put non-violent video clips here
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full demo (no dataset needed)
```bash
python demo.py
```
This generates synthetic training data, trains the model, and shows evaluation metrics.

### 3. Train on your own dataset
Place your `.mp4` / `.avi` video files in:
- `data/violence/` → violent clips
- `data/non_violence/` → non-violent clips

Then run:
```bash
python train.py
```

### 4. Predict on a video file
```bash
python predict.py --video path/to/your/video.mp4
```

### 5. Real-time webcam prediction
```bash
python predict.py --webcam
```

### 6. Evaluate model performance
```bash
python evaluate.py
```

## 🧠 How It Works

```
Video Frames
    ↓
Optical Flow (Farneback)    ← Motion magnitude & direction
    ↓
HOG Features               ← Shape/texture descriptors
    ↓
Feature Vector             ← Combined per-clip feature
    ↓
SVM / Random Forest        ← Binary classifier
    ↓
Violence / Non-Violence
```

### Key Features:
- **Optical Flow**: Captures motion patterns (fight = fast chaotic motion)
- **HOG Descriptors**: Captures pose/shape information per frame
- **Temporal Aggregation**: Statistics (mean, std, max) across all frames
- **Ensemble**: SVM + Random Forest with voting
- **Sliding Window**: Frame-by-frame real-time inference

## 📊 Feature Engineering

| Feature Group | Description |
|---|---|
| Flow Magnitude Stats | Mean, std, max, min of motion strength |
| Flow Direction Stats | Histogram of motion angles |
| HOG Features | Histogram of Oriented Gradients |
| Motion Energy | Overall scene motion energy |
| Chaos Score | Variance of flow vectors (fights = high chaos) |

## 📦 Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- scikit-learn
- numpy
- matplotlib
- scipy

## 🎯 Accuracy Notes
With real labeled datasets (e.g., RWF-2000, Hockey Fight Dataset), this approach typically achieves **85–92% accuracy**.

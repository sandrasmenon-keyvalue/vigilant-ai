# Vigilant AI - Drowsiness Detection System

A real-time drowsiness detection system that analyzes video frames at 5 FPS and outputs a drowsiness score (0-1).

## 🎯 Overview

This system follows a 7-step approach:
1. **Data Preparation**: Extract frames at 5 FPS from video datasets
2. **Face Detection**: Use MediaPipe to find faces and facial landmarks
3. **Feature Calculation**: Compute EAR, MAR, blink frequency, and head nods
4. **Window Creation**: Aggregate features over 5-second sliding windows
5. **Model Training**: Train XGBoost on extracted features
6. **Real-time Prediction**: Process live video streams
7. **Score Smoothing**: Apply temporal smoothing for stable output

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python ai-module/demo.py
```

## 📁 Project Structure

```
ai-module/
├── data_preparation.py    # Step 1: Frame extraction
├── face_detection.py      # Step 2: MediaPipe landmarks
├── feature_extraction.py  # Step 3: EAR, MAR, etc.
├── window_processing.py   # Step 4: Sliding windows
├── model_training.py      # Step 5: XGBoost training
├── realtime_detection.py  # Step 6: Live prediction
├── utils.py              # Helper functions
└── demo.py               # Complete demo
```

## 🎮 Usage

The system outputs a drowsiness score between 0 (alert) and 1 (drowsy) every second.

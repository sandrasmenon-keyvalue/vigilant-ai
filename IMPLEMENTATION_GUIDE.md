# 🚗 Vigilant AI - Drowsiness Detection Implementation Guide

## 🎯 System Overview

This implementation follows your **7-step plan** exactly:

1. **Data Preparation** → Extract frames at 5 FPS
2. **Face Detection** → MediaPipe landmarks (eyes, mouth, head)
3. **Feature Calculation** → EAR, MAR, blink frequency, head nods
4. **Window Processing** → 5-second sliding windows with aggregation
5. **Model Training** → XGBoost on extracted features
6. **Real-time Detection** → Live video processing
7. **Score Smoothing** → Temporal smoothing for stable output

**End Result**: Every second, output a **Drowsiness Score (0-1)**

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Demo
```bash
python ai-module/demo.py
```

### 3. Train on Real Data
```bash
python ai-module/train_model.py --dataset /path/to/your/dataset
```

---

## 📁 Project Structure

```
vigilant-ai/
├── requirements.txt              # Dependencies
├── README.md                    # Project overview
├── IMPLEMENTATION_GUIDE.md      # This guide
└── ai-module/
    ├── __init__.py
    ├── data_preparation.py      # Step 1: Frame extraction
    ├── face_detection.py        # Step 2: MediaPipe landmarks
    ├── feature_extraction.py    # Step 3: EAR, MAR, blinks, nods
    ├── window_processing.py     # Step 4: 5-second windows
    ├── model_training.py        # Step 5: XGBoost training
    ├── realtime_detection.py    # Step 6: Live detection
    ├── utils.py                 # Helper functions
    ├── demo.py                  # Complete demo
    └── train_model.py           # Training script
```

---

## 🔧 Step-by-Step Implementation

### Step 1: Data Preparation (`data_preparation.py`)
- **Input**: Video files (MP4, AVI, etc.)
- **Process**: Extract frames at 5 FPS
- **Output**: Frame images + metadata CSV

```python
extractor = FrameExtractor(target_fps=5)
frame_df = extractor.extract_frames_from_dataset("videos/", "frames/")
```

### Step 2: Face Detection (`face_detection.py`)
- **Input**: Frame images
- **Process**: MediaPipe face mesh detection
- **Output**: 468 facial landmarks per face

```python
detector = FaceLandmarkDetector()
landmarks = detector.detect_landmarks(frame)
```

### Step 3: Feature Calculation (`feature_extraction.py`)
- **Input**: Facial landmarks
- **Process**: Calculate drowsiness indicators
- **Output**: Feature dictionary

**Key Features:**
- **EAR (Eye Aspect Ratio)**: `(|p2-p6| + |p3-p5|) / (2 * |p1-p4|)`
- **MAR (Mouth Aspect Ratio)**: Similar formula for mouth
- **Blink Frequency**: Blinks per minute
- **Head Nods**: Sudden pitch changes
- **Eye Closure**: Percentage of closed eyes

```python
extractor = DrowsinessFeatureExtractor()
features = extractor.extract_features(landmarks, timestamp)
```

### Step 4: Window Processing (`window_processing.py`)
- **Input**: Individual frame features
- **Process**: 5-second sliding windows
- **Output**: Aggregated window features

**Aggregations:**
- Mean, std, min, max for each feature
- Blink/nod counts and rates
- Eye closure patterns
- Head movement amplitude

```python
processor = SlidingWindowProcessor(window_size_seconds=5.0, fps=5.0)
window_features = processor.add_feature(features)
```

### Step 5: Model Training (`model_training.py`)
- **Input**: Window features + labels
- **Process**: Train XGBoost classifier
- **Output**: Trained model + scaler

```python
trainer = DrowsinessModelTrainer(model_type='xgboost')
results = trainer.train_model(X, y, feature_names)
```

### Step 6: Real-time Detection (`realtime_detection.py`)
- **Input**: Live video stream
- **Process**: Full pipeline at 5 FPS
- **Output**: Drowsiness score every second

```python
detector = RealTimeDrowsinessDetector(model_path, scaler_path)
detector.process_video_stream(video_source=0)
```

### Step 7: Score Smoothing (`realtime_detection.py`)
- **Input**: Raw prediction scores
- **Process**: Exponential smoothing
- **Output**: Stable, smoothed scores

```python
smoothed_score = alpha * new_score + (1 - alpha) * previous_score
```

---

## 📊 Key Features Explained

### Eye Aspect Ratio (EAR)
```python
def calculate_ear(eye_landmarks):
    # Vertical distances
    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
    # Horizontal distance
    horizontal = distance(p1, p4)
    # EAR formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear
```

**Interpretation:**
- EAR > 0.25: Eyes open (alert)
- EAR < 0.25: Eyes closed (drowsy)

### Mouth Aspect Ratio (MAR)
```python
def calculate_mar(mouth_landmarks):
    # Similar to EAR but for mouth
    mar = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return mar
```

**Interpretation:**
- MAR > 0.5: Mouth open (yawning)
- MAR < 0.5: Mouth closed (normal)

### Blink Detection
```python
def detect_blink(ear_values):
    # Blink: EAR drops below threshold then rises
    if (ear[-3] > threshold and 
        ear[-2] < threshold and 
        ear[-1] > threshold):
        return True
```

---

## 🎮 Usage Examples

### Demo Mode
```bash
# Complete 7-step demo
python ai-module/demo.py
# Choose option 1 for complete demo
```

### Training Mode
```bash
# Train on your dataset
python ai-module/train_model.py \
    --dataset /path/to/videos \
    --output trained_models \
    --model xgboost
```

### Real-time Mode
```python
from ai_module.realtime_detection import RealTimeDrowsinessDetector

detector = RealTimeDrowsinessDetector("model.pkl", "scaler.pkl")
detector.process_video_stream(video_source=0)  # Webcam
```

---

## 📈 Expected Performance

### Accuracy
- **Synthetic Data**: ~85-90% accuracy
- **Real Data**: Depends on data quality
- **Production**: 80-95% with good training data

### Speed
- **Frame Processing**: ~50-100ms per frame
- **Real-time Capability**: 5-10 FPS
- **Memory Usage**: ~100-200MB

### Output
- **Score Range**: 0.0 (alert) to 1.0 (drowsy)
- **Update Frequency**: Every second
- **Smoothing**: 0.3 alpha (configurable)

---

## 🔧 Customization

### Adjust Thresholds
```python
extractor = DrowsinessFeatureExtractor(
    ear_threshold=0.25,    # Eye closure threshold
    mar_threshold=0.5      # Yawn threshold
)
```

### Change Window Size
```python
processor = SlidingWindowProcessor(
    window_size_seconds=5.0,  # Analysis window
    fps=5.0                   # Frame rate
)
```

### Modify Smoothing
```python
detector = RealTimeDrowsinessDetector(...)
detector.smoothing_alpha = 0.3  # 0.1 = more smoothing, 0.5 = less
```

---

## 🚨 Production Considerations

### Data Requirements
- **Minimum**: 100+ videos per class (drowsy/alert)
- **Recommended**: 500+ videos per class
- **Quality**: Good lighting, clear face visibility
- **Duration**: 10-30 seconds per video

### Hardware Requirements
- **CPU**: Modern multi-core processor
- **RAM**: 4GB+ recommended
- **Camera**: 720p+ webcam or IP camera
- **GPU**: Optional, for faster processing

### Integration
```python
# Get current status
status = detector.get_current_status()
print(f"Score: {status['drowsiness_score']:.3f}")
print(f"Drowsy: {status['is_drowsy']}")
print(f"Confidence: {status['confidence']:.3f}")
```

---

## 🎯 Next Steps

1. **Collect Real Data**: Record drowsy/alert driving videos
2. **Train Model**: Use `train_model.py` with your data
3. **Test Performance**: Validate on unseen data
4. **Deploy**: Integrate with your application
5. **Monitor**: Track performance in production

---

## 🆘 Troubleshooting

### Common Issues

**No face detected:**
- Check lighting conditions
- Ensure face is clearly visible
- Verify MediaPipe installation

**Low accuracy:**
- Collect more training data
- Check data quality and labeling
- Try different model types

**Slow performance:**
- Reduce frame resolution
- Lower FPS target
- Use GPU acceleration

**Unstable scores:**
- Increase smoothing alpha
- Check for camera shake
- Verify landmark detection quality

---

## 📚 References

- **MediaPipe**: Google's face mesh detection
- **XGBoost**: Gradient boosting for classification
- **EAR/MAR**: Eye/Mouth Aspect Ratio formulas
- **OpenCV**: Computer vision library

---

**🎉 You now have a complete drowsiness detection system following your exact 7-step plan!**

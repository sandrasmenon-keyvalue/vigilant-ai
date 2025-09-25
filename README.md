# Vigilant AI - Drowsiness Detection System

A real-time drowsiness detection system that analyzes images/frames at 5 FPS and outputs a drowsiness score (0-1).

## 📋 Requirements

- **Python**: 3.11.13
- **Dependencies**: See `requirements.txt`

## 🎯 Overview

This system follows a 5-step approach:
1. **Face Detection**: Use MediaPipe to find faces and facial landmarks
2. **Feature Calculation**: Compute EAR, MAR, blink frequency, and head nods
3. **Model Training**: Train XGBoost on extracted features from images
4. **Image Inference**: Process individual images for drowsiness detection
5. **Score Output**: Output drowsiness score (0-1) per image

### 🔄 Training vs Inference

**Training Phase:**
- Input: Images/frames in drowsy/ and not_drowsy/ folders
- Process: Extract features from images → Train model
- Output: Trained model

**Inference Phase:**
- Input: Images/frames (single image or batch)
- Process: Extract features → Predict drowsiness
- Output: Drowsiness score (0-1) per image

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
└── vision-model/
    ├── face_detection.py              # Step 1: MediaPipe landmarks
    ├── feature_extraction.py          # Step 2: EAR, MAR, etc.
    ├── window_processing.py           # Step 3: Feature aggregation
    ├── model_training.py              # Step 4: XGBoost training
    ├── utils.py                      # Helper functions
    ├── demo.py                       # Complete demo
    ├── train_model.py                # Training script
    ├── train_from_images.py          # Image-based training pipeline
    ├── setup_resources.py            # Setup resources directory
    ├── inference_single_image.py     # Single image inference
    ├── health_score.py               # Health score calculation (0.6*dv + 0.4*hv)
    └── integrated_health_analysis.py # Combined drowsiness + health analysis
```

## 🎮 Usage

### Training
```bash
# Setup resources directory structure
python ai-module/vision-model/setup_resources.py --resources ./resources

# Train model from images in resources folder
python ai-module/vision-model/train_from_images.py --resources ./resources --output ./training_output
```

### Inference
```bash
# Process single image
python ai-module/vision-model/inference_single_image.py --model ./training_output/models/drowsiness_model.pkl --scaler ./training_output/models/feature_scaler.pkl --image image.jpg

# Process directory of images
python ai-module/vision-model/inference_single_image.py --model ./training_output/models/drowsiness_model.pkl --scaler ./training_output/models/feature_scaler.pkl --directory ./test_images --output results.json
```

### Health Score Analysis
```bash
# Demo health score calculation
python ai-module/vision-model/health_demo.py

# Analyze image sequence with health metrics (5 frames/second + 1 health metric/second)
python ai-module/vision-model/integrated_health_analysis.py \
    --model ./training_output/models/drowsiness_model.pkl \
    --scaler ./training_output/models/feature_scaler.pkl \
    --images frame1.jpg frame2.jpg frame3.jpg frame4.jpg frame5.jpg \
    --health-metrics sample_health_metrics.json \
    --output health_analysis.json

# Analyze directory with health metrics
python ai-module/vision-model/integrated_health_analysis.py \
    --model ./training_output/models/drowsiness_model.pkl \
    --scaler ./training_output/models/feature_scaler.pkl \
    --directory ./test_images \
    --health-metrics sample_health_metrics.json \
    --output health_analysis.json

# Custom weights (default: 0.6*dv + 0.4*hv)
python ai-module/vision-model/integrated_health_analysis.py \
    --model ./training_output/models/drowsiness_model.pkl \
    --scaler ./training_output/models/feature_scaler.pkl \
    --images frame1.jpg frame2.jpg frame3.jpg frame4.jpg frame5.jpg \
    --health-metrics sample_health_metrics.json \
    --dv-weight 0.7 --hv-weight 0.3
```

The system outputs a drowsiness score between 0 (not drowsy) and 1 (drowsy) per image.

## 🏥 Health Score System

The health score system combines drowsiness detection with health metrics:

### **Data Structure:**
- **DV (Drowsiness Value)**: Calculated from 5 frames per second (drowsiness rate from 5 images)
- **HV (Health Value)**: Calculated from health metrics per second (heart rate, SpO2, age, health condition)

### **Health Score Formula:**
```
Health Score = 0.6 * DV + 0.4 * HV
```

### **Health Metrics Scoring:**
- **Heart Rate**: 60-100 BPM = optimal (1.0), outside range = scaled down
- **SpO2**: 95-100% = optimal (1.0), below 90% = critical (0.0)
- **Age**: Younger = better (30 years = 1.0, older = gradual decline)
- **Health Condition**: excellent(1.0) > good(0.8) > fair(0.6) > poor(0.4) > critical(0.2)

### **Health Score Interpretations:**
- **0.8-1.0**: Excellent Health
- **0.6-0.8**: Good Health  
- **0.4-0.6**: Fair Health
- **0.2-0.4**: Poor Health
- **0.0-0.2**: Critical Health

### **Health Metrics JSON Format:**
```json
[
  {
    "heart_rate": 75,
    "spo2": 98,
    "age": 35,
    "health_condition": "good"
  }
]
```

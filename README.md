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
    ├── face_detection.py          # Step 1: MediaPipe landmarks
    ├── feature_extraction.py      # Step 2: EAR, MAR, etc.
    ├── window_processing.py       # Step 3: Feature aggregation
    ├── model_training.py          # Step 4: XGBoost training
    ├── utils.py                  # Helper functions
    ├── demo.py                   # Complete demo
    ├── train_model.py            # Training script
    ├── train_from_images.py      # Image-based training pipeline
    ├── setup_resources.py        # Setup resources directory
    └── inference_single_image.py # Single image inference
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

The system outputs a drowsiness score between 0 (not drowsy) and 1 (drowsy) per image.

# Health Score Calculator

This module calculates a comprehensive health score using the formula: **0.6 × DV + 0.4 × HV**

Where:
- **DV (Drowsiness Score)**: Calculated from facial features using the vision model
- **HV (Health Variability Score)**: Calculated from vital signs using the vitals model

## Features

- **Real-time Processing**: Processes images at 5 FPS for continuous monitoring
- **Multi-modal Analysis**: Combines computer vision and vital signs data
- **Temporal Analysis**: Tracks health trends over time
- **Comprehensive Features**: Extracts EAR, MAR, blink frequency, head nods, and more
- **Health Interpretation**: Provides risk levels and recommendations

## Components

### Vision Model (DV Score)
Extracts and analyzes facial features:
- **Eye Aspect Ratio (EAR)**: Detects eye closure
- **Mouth Aspect Ratio (MAR)**: Detects yawning
- **Blink Frequency**: Tracks blinking patterns
- **Head Pose**: Detects nodding and head movements
- **Facial Landmarks**: Uses 68-point facial landmark detection

### Vitals Model (HV Score)
Analyzes vital signs and health conditions:
- **Heart Rate (HR)**: Beats per minute
- **SpO2**: Blood oxygen saturation
- **Age**: Patient age
- **Health Conditions**: Diabetes, hypertension, heart disease, etc.

## Usage

### Basic Usage

```python
from health_score_calculation import HealthScoreCalculator

# Initialize calculator
calculator = HealthScoreCalculator()

# Update vitals data
calculator.update_vitals(hr=75, spo2=98, age=30, health_condition=0)

# Calculate health score from image
result = calculator.calculate_health_score(image)

print(f"Health Score: {result['health_score']:.3f}")
print(f"Status: {result['status']}")
```

### Real-time Processing

```python
import cv2

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        result = calculator.calculate_health_score(frame)
        # Display results
        display_results_on_frame(frame, result)
        cv2.imshow('Health Monitor', frame)
```

## Health Score Interpretation

| Score Range | Status | Description |
|-------------|--------|-------------|
| 0.0 - 0.3 | Excellent | Low risk, good health indicators |
| 0.3 - 0.5 | Good | Moderate risk, some concerns |
| 0.5 - 0.7 | Moderate | High risk, multiple concerns |
| 0.7 - 1.0 | Poor | Critical risk, immediate attention needed |

## Model Files

The calculator uses the following trained models:

### Vision Model
- **Model**: `trained_models/vision-training/drowsiness_model.pkl`
- **Scaler**: `trained_models/vision-training/feature_scaler.pkl`
- **Features**: 14 facial features (EAR, MAR, head pose, etc.)

### Vitals Model
- **Model**: `trained_models/vital-training/vitals_hv_model_xgboost.pkl`
- **Scaler**: `trained_models/vital-training/vitals_feature_scaler_xgboost.pkl`
- **Features**: 8 vital features (age, HR, SpO2, health conditions)

## Example Scripts

### 1. Webcam Demo
```bash
python example_usage.py
# Choose option 1 for real-time webcam processing
```

### 2. Single Image Processing
```bash
python example_usage.py
# Choose option 2 and provide image path
```

### 3. Batch Processing
```bash
python example_usage.py
# Choose option 3 and provide directory path
```

## API Reference

### HealthScoreCalculator Class

#### `__init__(vision_model_path, vision_scaler_path, vitals_model_path, vitals_scaler_path)`
Initialize the calculator with model paths.

#### `update_vitals(hr, spo2, age, health_condition)`
Update current vital signs data.

#### `calculate_health_score(image, timestamp)`
Calculate overall health score from image and current vitals.

#### `calculate_dv_score(image, timestamp)`
Calculate drowsiness score from image using vision model.

#### `calculate_hv_score()`
Calculate health variability score from current vitals.

#### `get_health_trends(window_size)`
Analyze health score trends over recent window.

#### `reset_counters()`
Reset all counters and buffers.

## Dependencies

- OpenCV (`cv2`)
- NumPy
- scikit-learn
- XGBoost
- joblib
- MediaPipe (for facial landmark detection)

## Performance

- **Target FPS**: 5 frames per second
- **Processing Time**: ~200ms per frame
- **Memory Usage**: ~50MB for models
- **Accuracy**: Depends on trained model performance

## Error Handling

The calculator includes comprehensive error handling:
- Missing face detection
- Model loading failures
- Invalid input data
- Processing timeouts

## Future Enhancements

- [ ] Real-time vital signs integration
- [ ] Multi-person detection
- [ ] Advanced temporal analysis
- [ ] Cloud-based processing
- [ ] Mobile app integration
- [ ] Custom model training interface

# Inference API Integration Guide

This guide shows how to integrate `drowsiness_score` and `timestamp` from `process_frame_inference` in `inference_api.py` into the `synchronized_inference_engine` for unified processing.

## Overview

The integration allows you to:
- Send drowsiness scores from vision processing to the synchronized inference engine
- Synchronize DV (vision) data with HV (vitals) data by timestamp
- Trigger alerts based on environmental conditions (temperature, CO2) just like existing alerts
- Calculate unified health scores combining vision and vitals data

## Integration Points

### 1. DV Data Input Method

The synchronized inference engine now has a dedicated method for receiving data from `inference_api.py`:

```python
def receive_dv_from_inference_api(self, drowsiness_score: float, timestamp: float, 
                                features: Dict[str, float] = None, 
                                alert_level: str = None, 
                                confidence: float = None,
                                frame_id: str = None) -> bool:
```

### 2. Data Flow

```
inference_api.py (process_frame_inference)
    ↓
drowsiness_score + timestamp
    ↓
synchronized_inference_engine.receive_dv_from_inference_api()
    ↓
Buffer and synchronize with HV data
    ↓
Calculate health scores and trigger alerts
```

## Usage Examples

### Basic Integration

```python
from inference.synchronized_inference_engine import SynchronizedInferenceEngine
from inference_api import process_frame_inference

# Initialize the synchronized inference engine
engine = SynchronizedInferenceEngine(
    health_score_callback=my_health_callback,
    alert_callback=my_alert_callback
)

# Process frame with inference_api
result = process_frame_inference(image, timestamp)

# Send to synchronized inference engine
engine.receive_dv_from_inference_api(
    drowsiness_score=result.drowsiness_score,
    timestamp=result.timestamp,
    features=result.features,
    alert_level=result.alert_level,
    confidence=result.confidence,
    frame_id=result.frame_id
)
```

### With Vitals Data

```python
# Send DV data from vision processing
engine.receive_dv_from_inference_api(
    drowsiness_score=0.75,
    timestamp=time.time(),
    alert_level='high'
)

# Send HV data from vitals processing
engine.receive_hv_data(
    hv_data={
        'hr': 85,
        'spo2': 94,
        'temperature': 28.0,
        'co2_level': 1200
    },
    timestamp=time.time() + 0.1,
    source='vitals_websocket'
)

# The engine will automatically:
# 1. Synchronize the data by timestamp
# 2. Calculate health scores
# 3. Check environmental conditions
# 4. Trigger alerts if needed
```

## Alert System Integration

Environmental alerts are now fully integrated with the existing alert system:

### Alert Types

- **DV Alerts**: `dv_high` - Driver appears drowsy
- **HV Alerts**: `hv_low` - Health indicators concerning
- **HR Alerts**: `hr_violations` - Heart rate abnormal
- **SpO2 Alerts**: `spo2_violations` - Oxygen level low
- **Environmental Alerts**:
  - `temperature_critical` - Critical temperature conditions
  - `temperature_warning` - Temperature warnings
  - `co2_dangerous` - Dangerous CO2 levels
  - `co2_concerning` - Concerning CO2 levels
  - `co2_elevated` - Elevated CO2 levels
  - `environmental_critical` - Overall critical environmental conditions

### Alert Callback

```python
def handle_alert(alert: Alert, alert_type: str):
    print(f"Alert: {alert_type} - {alert.reason}")
    
    # Handle different alert types
    if alert_type.startswith('environmental'):
        # Handle environmental alerts
        handle_environmental_alert(alert, alert_type)
    elif alert_type.startswith('dv'):
        # Handle drowsiness alerts
        handle_drowsiness_alert(alert, alert_type)
    # ... etc
```

## Data Structures

### DV Data from inference_api

```python
{
    'drowsiness_score': 0.75,      # 0-1 drowsiness probability
    'features': {                  # Extracted features
        'ear': 0.18,
        'mar': 0.25,
        'blink_freq': 0.6
    },
    'alert_level': 'high',         # low/medium/high/critical
    'confidence': 0.89,            # Prediction confidence
    'frame_id': 'frame_001',       # Frame identifier
    'source': 'inference_api'      # Data source
}
```

### Synchronized Result

```python
SynchronizedResult(
    timestamp=1703123456.789,
    dv_score=0.75,                 # Drowsiness score
    hv_score=0.65,                 # Health vitals score
    health_score=0.70,             # Combined health score
    dv_data={...},                 # Original DV data
    hv_data={...},                 # Original HV data
    sync_tolerance=0.05,           # Time difference between DV/HV
    processing_time=0.012          # Processing time in seconds
)
```

## Configuration

### Environmental Thresholds

```python
# Temperature thresholds (Celsius)
temperature_thresholds = {
    'min_safe': 18.0,
    'max_safe': 25.0,
    'critical_high': 30.0,
    'critical_low': 10.0
}

# CO2 thresholds (PPM)
co2_thresholds = {
    'normal': 400,
    'acceptable': 1000,
    'concerning': 2000,
    'dangerous': 5000
}
```

### Synchronization Settings

```python
engine = SynchronizedInferenceEngine(
    sync_tolerance=0.1,        # Max time difference for sync (seconds)
    max_buffer_size=1000,      # Max data points to buffer
    enable_logging=True        # Enable detailed logging
)
```

## Complete Integration Example

See `inference_api_integration_example.py` for a complete working example that demonstrates:

1. Receiving DV data from inference_api
2. Synchronizing with HV data
3. Calculating health scores
4. Triggering environmental alerts
5. Handling all types of alerts

## Benefits

- **Unified Processing**: All data (vision, vitals, environmental) processed together
- **Automatic Synchronization**: Data synchronized by timestamp automatically
- **Comprehensive Alerts**: Environmental alerts integrated with existing alert system
- **Real-time Processing**: Immediate processing and alert generation
- **Flexible Integration**: Easy to integrate with existing inference_api.py code

The system now provides complete monitoring of driver drowsiness, health vitals, and environmental conditions with unified alerting and health score calculation!

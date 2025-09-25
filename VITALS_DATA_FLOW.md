# Vitals Data Flow - Complete System Overview

## 1. WebSocket Data Reception

### Input Payload (from your backend):
```json
{
  "type": "inference.vital_info",
  "userId": "test-user-1",
  "payload": {
    "heartbeatBpm": 83,
    "spo2": 94,
    "temperatureC": 36,
    "co2Level": 45,
    "timestampMs": 1758837688272
  }
}
```

**Note**: Only messages with `type: "inference.vital_info"` are processed. Other event types are ignored.

### Processing in `websocket_client.py`:
1. **Raw Message Reception**: WebSocket receives JSON string
2. **JSON Parsing**: `json.loads(message)` converts to Python dict
3. **Event Type Filtering**: Only processes messages with `type: "inference.vital_info"`
4. **Nested Payload Extraction**: Checks for `payload` field and extracts vitals data
5. **Field Mapping**: Converts your field names to internal format:
   - `heartbeatBpm` → `hr`
   - `spo2` → `spo2` (unchanged)
   - `temperatureC` → `temperature`
   - `co2Level` → `co2_level`
   - `timestampMs` → `timestamp` (converted from ms to seconds)
6. **Data Validation**: Ensures HR > 0 and SpO2 > 0
7. **Callback Trigger**: Calls `data_callback` with processed vitals

### Output Format (internal):
```json
{
  "timestamp": 1758837688.272,
  "hr": 83.0,
  "spo2": 94.0,
  "quality": "unknown",
  "temperature": 36.0,
  "co2_level": 45.0
}
```

## 2. Vitals Processing Pipeline

### In `vitals_websocket_processor.py`:

#### A. Raw Vitals Callback
- **Function**: `_process_vitals_data()`
- **Action**: Immediately forwards raw vitals to `handle_vitals_data()` in integrated service
- **Purpose**: Real-time data streaming to synchronized inference engine

#### B. HV Prediction Processing
- **Function**: `process_websocket_vitals()`
- **Input**: Raw vitals + age + health conditions
- **Processing**:
  1. **Feature Preparation**: Creates model input with:
     - `age`: Patient age
     - `hr_median`: Heart rate
     - `spo2_median`: Blood oxygen
     - Health conditions flags (diabetes, hypertension, etc.)
  2. **Model Inference**: Uses trained XGBoost model to predict HV score
  3. **Environmental Analysis**: Checks temperature and CO2 levels
  4. **Risk Assessment**: Categorizes risk level (low/medium/high/critical)

#### C. Output Generation
```json
{
  "input_data": {...},
  "hv_score": 0.75,
  "risk_level": "medium",
  "interpretation": "Elevated risk detected",
  "recommendations": ["Monitor closely", "Consider medical attention"],
  "timestamp": 1758837688.272,
  "environmental_conditions": {
    "temperature": {"value": 36, "status": "high", "alert": "High temperature: 36°C"},
    "co2_level": {"value": 45, "status": "normal"},
    "overall_risk": "moderate",
    "alerts": ["High temperature: 36°C"]
  },
  "original_vitals": {...}
}
```

## 3. Integrated System Processing

### In `integrated_live_stream_service.py`:

#### A. Raw Vitals Handler
- **Function**: `handle_vitals_data()`
- **Action**: Sends vitals to synchronized inference engine
- **Purpose**: Combines with drowsiness detection for integrated monitoring

#### B. Synchronized Inference Engine
- **Component**: `SynchronizedInferenceEngine`
- **Input**: HV data (vitals) + DV data (drowsiness from camera)
- **Processing**:
  1. **Data Synchronization**: Aligns vitals and drowsiness data by timestamp
  2. **Health Score Calculation**: Combines HV and DV scores
  3. **Alert Generation**: Triggers alerts based on combined risk
  4. **Callback Execution**: Calls health score and alert callbacks

#### C. Health Score Callback
- **Function**: `handle_health_score()`
- **Output**: Logs combined health metrics
- **Data**: DV score, HV score, overall health score

#### D. Alert Callback
- **Function**: `handle_alert()`
- **Action**: Broadcasts alerts to all active WebSocket streams
- **Format**:
```json
{
  "type": "integrated_alert",
  "alert_type": "health_vitals",
  "alert_level": "high",
  "reason": "Elevated heart rate and drowsiness detected",
  "timestamp": "2025-01-26T03:31:28.273Z",
  "stream_id": "stream-uuid"
}
```

## 4. Real-time Monitoring & Alerting

### WebSocket Stream Broadcasting
- **Target**: All connected video streams
- **Content**: Alerts, health scores, frame data
- **Format**: JSON messages over WebSocket

### Alert Types Generated
1. **Drowsiness Alerts**: Based on camera analysis
2. **Health Vitals Alerts**: Based on HR/SpO2 analysis
3. **Environmental Alerts**: Based on temperature/CO2
4. **Integrated Alerts**: Combined risk assessment

## 5. Complete Data Flow Summary

```
Backend WebSocket
    ↓ (JSON payload)
WebSocket Client
    ↓ (field mapping & validation)
Vitals WebSocket Processor
    ↓ (raw callback + HV prediction)
Integrated Live Stream Service
    ↓ (synchronized inference)
Synchronized Inference Engine
    ↓ (health score + alerts)
WebSocket Streams
    ↓ (real-time updates)
Frontend Clients
```

## 6. Key Components & Their Roles

### `websocket_client.py`
- **Role**: WebSocket communication layer
- **Responsibility**: Receive, parse, validate, and forward vitals data

### `vitals_websocket_processor.py`
- **Role**: Vitals processing orchestrator
- **Responsibility**: Coordinate raw data forwarding and HV prediction

### `vitals_processor.py`
- **Role**: ML model inference engine
- **Responsibility**: Predict health risk from vitals data

### `integrated_live_stream_service.py`
- **Role**: System integration hub
- **Responsibility**: Combine vitals with video analysis, manage alerts

### `synchronized_inference_engine.py`
- **Role**: Multi-modal inference coordinator
- **Responsibility**: Synchronize and combine different data streams

## 7. Configuration & Customization

### Patient Profile
- **Age**: Configurable (default: 35)
- **Health Conditions**: Diabetes, hypertension, heart disease, etc.
- **Environmental Thresholds**: Temperature and CO2 limits

### Alert Thresholds
- **Drowsiness**: 0.3 (medium), 0.5 (high), 0.7 (critical)
- **Health Vitals**: Based on ML model predictions
- **Environmental**: Temperature 18-25°C, CO2 < 1000 PPM

### WebSocket Endpoints
- **Vitals Input**: `ws://192.168.5.93:4000/ws?token=dev-shared-secret`
- **Video Stream**: `ws://localhost:8001/ws/integrated_stream`
- **Health Check**: `http://localhost:8001/health`

This system provides real-time, multi-modal health monitoring by combining physiological data (HR, SpO2) with behavioral data (drowsiness) and environmental data (temperature, CO2) for comprehensive risk assessment and alerting.

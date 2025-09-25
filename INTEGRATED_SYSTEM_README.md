# Integrated Vigilant AI System

Complete end-to-end integration of WebSocket, inference_api, synchronized_inference_engine, and vitals WebSocket for unified processing and alerting.

## 🎯 System Overview

The integrated system provides comprehensive monitoring of:
- **Driver Drowsiness** (DV) - from vision processing via inference_api.py
- **Health Vitals** (HV) - from WebSocket (HR, SpO2, temperature, CO2)
- **Environmental Conditions** - ambient temperature and CO2 levels
- **Unified Alerts** - all alert types integrated into single system
- **Health Scores** - combined DV and HV scores with environmental factors

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   WebSocket     │    │  inference_api   │    │  Vitals WebSocket   │
│   (Camera)      │───▶│  (DV Processing) │    │  (HV Data)          │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
         │                       │                       │
         │                       ▼                       ▼
         │              ┌─────────────────────────────────────┐
         │              │  Synchronized Inference Engine     │
         │              │  • DV + HV Synchronization         │
         │              │  • Environmental Monitoring        │
         │              │  • Health Score Calculation        │
         │              │  • Unified Alert System            │
         │              └─────────────────────────────────────┘
         │                                │
         ▼                                ▼
┌─────────────────┐              ┌─────────────────┐
│  Real-time      │              │  Alert System   │
│  WebSocket      │              │  • DV Alerts    │
│  Response       │              │  • HV Alerts    │
└─────────────────┘              │  • Environmental│
                                 │  • Unified      │
                                 └─────────────────┘
```

## 🚀 Quick Start

### 1. Start All Services

```bash
python start_integrated_system.py
```

This will start:
- Inference API (port 8002)
- Vitals WebSocket Server (port 8765) - simulated for testing
- Integrated Live Stream Service (port 8001)

### 2. Test the System

```bash
python test_integrated_system.py
```

### 3. Connect WebSocket Client

Connect to: `ws://localhost:8001/ws/integrated_stream`

Send configuration:
```json
{
  "stream_id": "my_stream_001",
  "camera_source": 0,
  "frame_width": 640,
  "frame_height": 480,
  "enable_display": true
}
```

## 📡 API Endpoints

### Integrated Live Stream Service (Port 8001)

- **Health Check**: `GET http://localhost:8001/health`
- **WebSocket**: `ws://localhost:8001/ws/integrated_stream`
- **API Docs**: `http://localhost:8001/docs`

### Inference API (Port 8002)

- **Health Check**: `GET http://localhost:8002/inference/health`
- **Single Frame**: `POST http://localhost:8002/inference/single_frame`
- **API Docs**: `http://localhost:8002/docs`

### Vitals WebSocket (Port 8765)

- **WebSocket**: `ws://localhost:8765`
- **Data Format**: JSON with HR, SpO2, temperature, CO2

## 🔄 Data Flow

### 1. DV Data Flow (Vision Processing)

```
Camera Frame → WebSocket → Integrated Service → inference_api.py
                                                      ↓
drowsiness_score + timestamp + features + alert_level
                                                      ↓
synchronized_inference_engine.receive_dv_from_inference_api()
```

### 2. HV Data Flow (Vitals Processing)

```
Vitals Backend → Vitals WebSocket → vitals_websocket_processor
                                                      ↓
HR + SpO2 + temperature + CO2 + timestamp
                                                      ↓
synchronized_inference_engine.receive_hv_data()
```

### 3. Synchronized Processing

```
DV Data + HV Data → Synchronized Inference Engine
                           ↓
Health Score Calculation + Environmental Monitoring
                           ↓
Unified Alert System (DV + HV + Environmental)
```

## 🚨 Alert System

### Alert Types

| Type | Trigger | Description |
|------|---------|-------------|
| `drowsiness` | DV score ≥ 0.5 | Driver appears drowsy |
| `dv_high` | DV score ≥ 0.7 | High drowsiness detected |
| `hv_low` | HV score < 0.3 | Health indicators concerning |
| `hr_violations` | HR outside 60-100 BPM | Heart rate abnormal |
| `spo2_violations` | SpO2 < 95% | Oxygen level low |
| `temperature_critical` | Temp < 10°C or > 30°C | Critical temperature |
| `temperature_warning` | Temp 10-18°C or 25-30°C | Temperature warning |
| `co2_dangerous` | CO2 > 5000 PPM | Dangerous CO2 level |
| `co2_concerning` | CO2 > 2000 PPM | Concerning CO2 level |
| `co2_elevated` | CO2 > 1000 PPM | Elevated CO2 level |
| `environmental_critical` | Overall critical conditions | Critical environmental |

### Alert Format

```json
{
  "type": "integrated_alert",
  "alert_type": "drowsiness",
  "alert_level": "high",
  "reason": "Driver appears drowsy (score: 0.75)",
  "drowsiness_score": 0.75,
  "timestamp": "2024-01-15T10:30:45.123Z",
  "stream_id": "my_stream_001"
}
```

## 🌡️ Environmental Monitoring

### Temperature Thresholds

| Status | Range | Action |
|--------|-------|--------|
| Critical Low | < 10°C | Immediate alert |
| Low | 10-18°C | Warning alert |
| Normal | 18-25°C | No alert |
| High | 25-30°C | Warning alert |
| Critical High | > 30°C | Immediate alert |

### CO2 Level Thresholds

| Status | Range | Action |
|--------|-------|--------|
| Normal | < 400 PPM | No alert |
| Acceptable | 400-1000 PPM | No alert |
| Elevated | 1000-2000 PPM | Warning alert |
| Concerning | 2000-5000 PPM | Alert |
| Dangerous | > 5000 PPM | Critical alert |

## 📊 Health Score Calculation

The system calculates unified health scores combining:

1. **DV Score** (0-1): Drowsiness probability from vision processing
2. **HV Score** (0-1): Health vitals score from HR, SpO2, age, conditions
3. **Environmental Factor**: Temperature and CO2 level adjustments
4. **Final Health Score**: Weighted combination of all factors

## 🔧 Configuration

### Synchronized Inference Engine

```python
synchronized_engine = SynchronizedInferenceEngine(
    sync_tolerance=0.1,        # Max time difference for sync (seconds)
    max_buffer_size=1000,      # Max data points to buffer
    health_score_callback=handle_health_score,
    alert_callback=handle_alert,
    enable_logging=True
)
```

### Vitals WebSocket Processor

```python
vitals_processor = VitalsWebSocketProcessor(
    websocket_uri="ws://localhost:8765",
    age=35,                    # Patient age
    health_conditions={        # Health conditions
        'diabetes': 0,
        'hypertension': 0,
        'heart_disease': 0,
        'respiratory_condition': 0,
        'smoker': 0
    }
)
```

## 🧪 Testing

### Test Individual Components

```bash
# Test inference API
python inference_api.py

# Test synchronized inference engine
python inference_api_integration_example.py

# Test vitals processor
python vitals_processing/run_vitals_processor.py
```

### Test Integrated System

```bash
# Start all services
python start_integrated_system.py

# Test end-to-end
python test_integrated_system.py
```

## 📁 File Structure

```
vigilant-ai/
├── integrated_live_stream_service.py    # Main integrated service
├── start_integrated_system.py           # Startup script
├── test_integrated_system.py            # Test script
├── inference/
│   └── synchronized_inference_engine.py # Synchronized processing
├── vitals_processing/
│   ├── vitals_processor.py              # Vitals processing
│   └── vitals_websocket_processor.py    # WebSocket integration
├── websocket/
│   └── websocket_client.py              # WebSocket client
└── inference_api.py                     # Vision processing API
```

## 🔍 Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check if all services are running
   - Verify ports are not blocked
   - Check firewall settings

2. **No DV Data Received**
   - Verify inference_api.py is running
   - Check camera access permissions
   - Verify WebSocket connection

3. **No HV Data Received**
   - Check vitals WebSocket connection
   - Verify data format matches expected schema
   - Check network connectivity

4. **Alerts Not Triggering**
   - Check alert thresholds configuration
   - Verify data synchronization
   - Check logging for errors

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🎉 Features

✅ **Complete Integration**: All components working together  
✅ **Real-time Processing**: Immediate data processing and alerts  
✅ **Environmental Monitoring**: Temperature and CO2 level monitoring  
✅ **Unified Alert System**: All alert types in single system  
✅ **Health Score Calculation**: Combined DV and HV scoring  
✅ **WebSocket Communication**: Real-time data streaming  
✅ **Configurable Thresholds**: Customizable alert levels  
✅ **Comprehensive Testing**: End-to-end test suite  
✅ **Detailed Logging**: Full visibility into system operation  
✅ **Error Handling**: Robust error recovery and reporting  

## 🚀 Next Steps

1. **Deploy to Production**: Configure for production environment
2. **Add Database**: Store alerts and health scores
3. **Mobile App**: Create mobile client for monitoring
4. **Analytics**: Add data analytics and reporting
5. **Machine Learning**: Enhance prediction accuracy
6. **Scalability**: Add load balancing and clustering

The integrated system is now ready for end-to-end testing and production deployment! 🎉

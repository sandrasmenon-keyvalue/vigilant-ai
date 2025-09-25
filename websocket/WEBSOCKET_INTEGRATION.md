# WebSocket Integration for HR and SpO2 Data

This module provides WebSocket client functionality to receive real-time HR and SpO2 data from your backend and process it using the trained vitals model.

## Files Created

- `websocket_client.py` - Core WebSocket client for receiving vitals data
- `vitals_websocket_processor.py` - Integrated processor combining WebSocket client with vitals analysis
- `run_vitals_processor.py` - Simple script to run the processor

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the processor:**
   ```bash
   python run_vitals_processor.py
   ```

3. **Customize connection:**
   ```bash
   python run_vitals_processor.py ws://your-backend:8765 45
   ```

## Expected WebSocket Message Format

Your backend should send JSON messages with this structure:
```json
{
  "timestamp": 1703123456.789,
  "hr": 75.5,
  "spo2": 98.2,
  "quality": "good"
}
```

## Features

- **Automatic Reconnection**: Handles connection drops and reconnects automatically
- **Real-time Processing**: Processes each HR/SpO2 reading for HV predictions
- **Configurable Callbacks**: Set custom handlers for data, predictions, and connection status
- **Health Conditions**: Configure patient health conditions for accurate predictions
- **Statistics Tracking**: Monitor message counts and processing statistics

## Usage Examples

### Basic Usage
```python
from vitals_websocket_processor import VitalsWebSocketProcessor

# Create processor
processor = VitalsWebSocketProcessor(
    websocket_uri="ws://localhost:8765",
    age=35,
    health_conditions={'hypertension': 1}
)

# Set callbacks
processor.set_hv_prediction_callback(my_prediction_handler)
processor.set_connection_status_callback(my_connection_handler)

# Start processing
await processor.start()
```

### Custom Callbacks
```python
def my_prediction_handler(hv_result):
    print(f"HV Score: {hv_result['hv_score']:.3f}")
    print(f"Risk Level: {hv_result['risk_level']}")

def my_connection_handler(is_connected):
    print(f"Connection: {'Connected' if is_connected else 'Disconnected'}")
```

## Configuration

- **WebSocket URI**: Default `ws://localhost:8765`
- **Patient Age**: Default 35 years
- **Health Conditions**: Configurable dictionary with conditions
- **Reconnection**: Automatic with configurable intervals

## Integration with Existing Code

The WebSocket processor integrates seamlessly with your existing `vitals_processor.py` and uses the same trained models for HV predictions.

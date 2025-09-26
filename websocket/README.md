# WebSocket Event Sender

Simple and reliable WebSocket client for sending events to your WebSocket server at `ws://192.168.5.93:4000/ws?token=dev-shared-secret`.

## Features

✅ **Automatic Health Score Sending**: Health scores are automatically sent via WebSocket after calculation in the inference engine  
✅ **Thread-Safe**: Works in any context (sync/async/threaded)  
✅ **Simple API**: Easy to use functions for sending events  
✅ **Auto-Connect**: Automatically connects when needed  
✅ **Error Handling**: Graceful error handling with logging  

## Quick Start

### Automatic Health Score Sending

Health scores are **automatically sent** when calculated in the `SynchronizedInferenceEngine`:

```python
from inference.synchronized_inference_engine import SynchronizedInferenceEngine

# Create inference engine (health scores will be sent automatically)
engine = SynchronizedInferenceEngine(
    sync_tolerance=0.1,
    enable_logging=True
)

# Send data - health scores will be automatically sent via WebSocket
engine.receive_dv_data({"drowsiness_score": 0.4}, timestamp, "vision")
engine.receive_hv_data({"hv_score": 0.8}, timestamp, "vitals")

# Health score is calculated and automatically sent via WebSocket!
```

### Manual Event Sending

You can also send events manually:

```python
from websocket.event_sender import send_health_score_sync, send_event_sync

# Send health score event
send_health_score_sync(
    health_score=0.75,
    dv_score=0.6,
    hv_score=0.9,
    mode="manual",
    source="my_app"
)

# Send custom event
send_event_sync("alert", {
    "message": "High drowsiness detected",
    "level": "warning",
    "action_required": True
})
```

### Async Usage

```python
from websocket.event_sender import send_health_score_event, send_event

# In async context
await send_health_score_event(0.68, dv_score=0.5, hv_score=0.8)
await send_event("status", {"system": "online", "uptime": 3600})
```

## Message Format

Health score messages sent to WebSocket server:

```json
{
  "type": "health_score",
  "timestamp": 1758855248.477,
  "health_score": 0.560,
  "dv_score": 0.400,
  "hv_score": 0.800,
  "interpretation": "Fair Health",
  "risk_level": "moderate",
  "mode": "synchronized",
  "sync_tolerance": 0.050,
  "source": "synchronized_inference_engine"
}
```

## API Reference

### Functions

#### `send_health_score_sync(health_score, dv_score=None, hv_score=None, **kwargs)`
Send health score event (works in any context).

#### `send_event_sync(event_type, data)`
Send any event (works in any context).

#### `send_health_score_event(health_score, dv_score=None, hv_score=None, **kwargs)` (async)
Send health score event (async version).

#### `send_event(event_type, data)` (async)
Send any event (async version).

### Classes

#### `WebSocketEventSender(uri="ws://192.168.5.93:4000/ws?token=dev-shared-secret")`
Main WebSocket client class.

**Methods:**
- `await connect()` - Connect to WebSocket server
- `await disconnect()` - Disconnect from server
- `await send_event(event_type, data)` - Send event
- `await send_health_score(health_score, ...)` - Send health score

## Configuration

The WebSocket URI is configured in `websocket/event_sender.py`:

```python
# Default URI
uri = "ws://192.168.5.93:4000/ws?token=dev-shared-secret"

# Custom URI
sender = WebSocketEventSender("ws://your-server:port/ws?token=your-token")
```

## Integration Status

✅ **Integrated with SynchronizedInferenceEngine**: Health scores are automatically sent  
✅ **Works with all processing modes**: Synchronized, DV-only, HV-only  
✅ **Production ready**: Used in `inference_api.py` and `live_stream_service.py`  

## Testing

Run the test to verify everything works:

```bash
python test_health_score_websocket_integration.py
```

## Troubleshooting

### Health scores not being sent?

1. **Check logs** for `📤 Health score sent via WebSocket` messages
2. **Verify WebSocket server** is running at `ws://192.168.5.93:4000/ws?token=dev-shared-secret`
3. **Check network connectivity** between client and server

### Connection issues?

1. **Verify URI** is correct in `websocket/event_sender.py`
2. **Check token** is valid (`dev-shared-secret`)
3. **Verify server** accepts connections

### No events received on server?

1. **Check server logs** for incoming WebSocket messages
2. **Verify server** handles `health_score` event type
3. **Test with** `python websocket/event_sender.py`

## Examples

### Production Usage

```python
# In your application
from inference.synchronized_inference_engine import SynchronizedInferenceEngine

# Health scores automatically sent via WebSocket
engine = SynchronizedInferenceEngine(enable_logging=True)

# Process your data
engine.receive_dv_data(vision_data, timestamp, "camera")
engine.receive_hv_data(vitals_data, timestamp, "sensor")

# Health score calculated and sent automatically!
```

### Custom Events

```python
from websocket.event_sender import send_event_sync

# Send alert
send_event_sync("alert", {
    "type": "drowsiness_warning",
    "level": "high",
    "message": "Driver appears drowsy",
    "timestamp": time.time(),
    "action": "suggest_break"
})

# Send status update
send_event_sync("status", {
    "system": "vigilant_ai",
    "status": "active",
    "uptime": 7200,
    "health_scores_sent": 150
})
```

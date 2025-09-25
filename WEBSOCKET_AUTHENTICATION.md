# WebSocket Authentication Setup

## 🔐 **Authentication Methods**

The WebSocket client now supports multiple authentication methods for connecting to your backend server.

### **Method 1: Environment Variables (Recommended)**

Set environment variables before running the system:

```bash
# Set API key
export VITALS_API_KEY="your-api-key-here"

# Set Bearer token
export VITALS_AUTH_TOKEN="your-bearer-token-here"

# Set WebSocket URI (optional)
export VITALS_WEBSOCKET_URI="wss://your-backend.com:8765"
```

### **Method 2: Direct Configuration**

Pass credentials directly when creating the client:

```python
from websocket.websocket_client import VitalsWebSocketClient

# With API key
client = VitalsWebSocketClient(
    uri="wss://your-backend.com:8765",
    api_key="your-api-key-here"
)

# With Bearer token
client = VitalsWebSocketClient(
    uri="wss://your-backend.com:8765",
    auth_token="your-bearer-token-here"
)

# With both
client = VitalsWebSocketClient(
    uri="wss://your-backend.com:8765",
    api_key="your-api-key-here",
    auth_token="your-bearer-token-here"
)
```

### **Method 3: Custom Headers**

For custom authentication schemes:

```python
client = VitalsWebSocketClient(
    uri="wss://your-backend.com:8765",
    headers={
        "X-API-Key": "your-api-key",
        "Authorization": "Bearer your-token",
        "X-Client-ID": "vigilant-ai-client",
        "X-User-ID": "user123"
    }
)
```

## 🚀 **Usage Examples**

### **Basic Usage with Environment Variables**

```bash
# Set credentials
export VITALS_API_KEY="abc123"
export VITALS_AUTH_TOKEN="xyz789"

# Run the system
python start_integrated_system.py
```

### **Programmatic Usage**

```python
from vitals_processing.vitals_websocket_processor import VitalsWebSocketProcessor

# Create processor with authentication
processor = VitalsWebSocketProcessor(
    websocket_uri="wss://your-backend.com:8765",
    api_key="your-api-key",
    auth_token="your-bearer-token"
)

# Start processing
await processor.start()
```

### **Dynamic Credential Updates**

```python
# Update credentials after creation
processor.websocket_client.set_credentials(
    api_key="new-api-key",
    auth_token="new-token"
)
```

## 🔧 **Configuration Files**

### **Create .env file (recommended):**

```bash
# .env
VITALS_API_KEY=your-api-key-here
VITALS_AUTH_TOKEN=your-bearer-token-here
VITALS_WEBSOCKET_URI=wss://your-backend.com:8765
```

### **Load from .env file:**

```python
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('VITALS_API_KEY')
auth_token = os.getenv('VITALS_AUTH_TOKEN')
websocket_uri = os.getenv('VITALS_WEBSOCKET_URI', 'ws://localhost:8765')
```

## 🛡️ **Security Best Practices**

1. **Never commit credentials to version control**
2. **Use environment variables for production**
3. **Rotate API keys regularly**
4. **Use HTTPS/WSS for production connections**
5. **Validate tokens on the server side**

## 🔍 **Troubleshooting**

### **Connection Failed - Authentication Error**

```
❌ Failed to connect to WebSocket server: 401 Unauthorized
```

**Solution:** Check your API key or token is correct and has proper permissions.

### **No Authentication Credentials**

```
⚠️ No authentication credentials provided - connection may fail if server requires auth
```

**Solution:** Set `VITALS_API_KEY` or `VITALS_AUTH_TOKEN` environment variables.

### **Invalid Headers**

```
❌ Failed to connect to WebSocket server: 400 Bad Request
```

**Solution:** Check your custom headers format and ensure they match server expectations.

## 📋 **Backend Server Requirements**

Your backend WebSocket server should:

1. **Accept authentication headers:**
   - `X-API-Key: your-api-key`
   - `Authorization: Bearer your-token`

2. **Validate credentials before allowing connection**

3. **Return appropriate error codes:**
   - `401 Unauthorized` for invalid credentials
   - `403 Forbidden` for insufficient permissions
   - `400 Bad Request` for malformed requests

## 🧪 **Testing Authentication**

```python
# Test connection with credentials
import asyncio
from websocket.websocket_client import VitalsWebSocketClient

async def test_auth():
    client = VitalsWebSocketClient(
        uri="wss://your-backend.com:8765",
        api_key="test-key"
    )
    
    try:
        await client.connect()
        print("✅ Authentication successful!")
        await client.disconnect()
    except Exception as e:
        print(f"❌ Authentication failed: {e}")

asyncio.run(test_auth())
```

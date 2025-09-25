"""
WebSocket Client for Receiving HR and SpO2 Data
Connects to backend WebSocket server to receive real-time vitals data.
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, Callable, Optional
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)


class VitalsWebSocketClient:
    """
    WebSocket client for receiving HR and SpO2 data from backend.
    """
    
    def __init__(self, 
                 uri: str = "ws://localhost:8765",
                 api_key: Optional[str] = None,
                 auth_token: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None,
                 reconnect_interval: int = 5,
                 max_reconnect_attempts: int = 10):
        """
        Initialize WebSocket client.
        
        Args:
            uri: WebSocket server URI
            api_key: API key for authentication (can also be set via VITALS_API_KEY env var)
            auth_token: Bearer token for authentication (can also be set via VITALS_AUTH_TOKEN env var)
            headers: Additional headers to send with connection
            reconnect_interval: Seconds to wait before reconnecting
            max_reconnect_attempts: Maximum number of reconnection attempts
        """
        self.uri = uri
        self.api_key = api_key or os.getenv('VITALS_API_KEY')
        self.auth_token = auth_token or os.getenv('VITALS_AUTH_TOKEN')
        self.headers = headers or {}
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.websocket = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.data_callback: Optional[Callable] = None
        self.connection_callback: Optional[Callable] = None
        self.disconnection_callback: Optional[Callable] = None
        
        # Build authentication headers
        self._build_auth_headers()
    
    def _build_auth_headers(self):
        """Build authentication headers for WebSocket connection."""
        auth_headers = {}
        
        # Add API key if provided
        if self.api_key:
            auth_headers['X-API-Key'] = self.api_key
            logger.debug("Added API key to headers")
        
        # Add Bearer token if provided
        if self.auth_token:
            auth_headers['Authorization'] = f'Bearer {self.auth_token}'
            logger.debug("Added Bearer token to headers")
        
        # Merge with custom headers
        self.headers.update(auth_headers)
        
        if self.headers:
            logger.info(f"WebSocket will connect with headers: {list(self.headers.keys())}")
        else:
            logger.warning("No authentication credentials provided - connection may fail if server requires auth")
    
    def set_credentials(self, api_key: Optional[str] = None, auth_token: Optional[str] = None):
        """
        Update authentication credentials.
        
        Args:
            api_key: New API key
            auth_token: New Bearer token
        """
        if api_key is not None:
            self.api_key = api_key
        if auth_token is not None:
            self.auth_token = auth_token
        
        # Rebuild headers with new credentials
        self._build_auth_headers()
        
    def set_data_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Set callback function to handle received vitals data.
        
        Args:
            callback: Function that receives vitals data dict with keys:
                     - timestamp: Unix timestamp (or timestampMs converted to seconds)
                     - hr: Heart rate in BPM (from heartbeatBpm)
                     - spo2: Blood oxygen saturation percentage
                     - quality: Data quality indicator (optional)
                     - temperature: Ambient temperature in Celsius (from temperatureC)
                     - co2_level: CO2 level in PPM (from co2Level)
        """
        self.data_callback = callback
    
    def set_connection_callback(self, callback: Callable[[], None]):
        """
        Set callback function for successful connection.
        """
        self.connection_callback = callback
    
    def set_disconnection_callback(self, callback: Callable[[], None]):
        """
        Set callback function for disconnection.
        """
        self.disconnection_callback = callback
    
    async def connect(self):
        """Connect to WebSocket server with authentication."""
        try:
            logger.info(f"Connecting to WebSocket server: {self.uri}")
            
            # Connect with authentication headers
            if self.headers:
                logger.debug(f"Using headers: {self.headers}")
                self.websocket = await websockets.connect(self.uri, extra_headers=list(self.headers.items()))
            else:
                self.websocket = await websockets.connect(self.uri)
            
            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info("✅ Connected to WebSocket server")
            
            if self.connection_callback:
                self.connection_callback()
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to WebSocket server: {e}")
            self.is_connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            logger.info("Disconnected from WebSocket server")
            
            if self.disconnection_callback:
                self.disconnection_callback()
    
    async def listen(self):
        """
        Listen for messages from WebSocket server.
        Processes HR and SpO2 data as it arrives.
        """
        if not self.websocket:
            raise RuntimeError("Not connected to WebSocket server")
        
        try:
            async for message in self.websocket:
                await self._process_message(message)
                
        except ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.is_connected = False
        except WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Unexpected error in listen loop: {e}")
            self.is_connected = False
    
    async def _process_message(self, message: str):
        """
        Process incoming WebSocket message.
        
        Args:
            message: JSON string containing vitals data in format:
                    {"type": "inference.vital_info", "payload": {"heartbeatBpm": 75, "spo2": 98, ...}}
                    Only processes messages with type="inference.vital_info"
        """
        try:
            # Parse JSON message
            data = json.loads(message)
            
            # Debug: Log the raw payload received
            logger.info(f"Raw WebSocket payload received: {data}")
            
            # Check if this is a vitals event
            event_type = data.get('type', '')
            if event_type != 'inference.vital_info':
                logger.debug(f"Ignoring non-vitals event: {event_type}")
                return
            
            # Extract vitals data - handle nested payload structure
            # Check if data is nested in a 'payload' field
            if 'payload' in data and isinstance(data['payload'], dict):
                payload_data = data['payload']
                logger.info(f"Found nested payload for {event_type}: {payload_data}")
            else:
                payload_data = data
            
            timestamp_value = payload_data.get('timestamp', payload_data.get('timestampMs', time.time()))
            # Convert timestampMs to seconds if it's in milliseconds
            if 'timestampMs' in payload_data and timestamp_value > 1e10:  # If timestamp is in milliseconds
                timestamp_value = timestamp_value / 1000.0
            
            vitals_data = {
                'timestamp': timestamp_value,
                'hr': float(payload_data.get('hr', payload_data.get('heartbeatBpm', 0))),
                'spo2': float(payload_data.get('spo2', payload_data.get('spo2', 0))),
                'quality': payload_data.get('quality', 'unknown'),
                'temperature': payload_data.get('temperature', payload_data.get('temperatureC', None)),  # Ambient temperature in Celsius
                'co2_level': payload_data.get('co2_level', payload_data.get('co2Level', None))      # CO2 level in PPM
            }
            
            # Debug: Log the processed vitals data
            logger.info(f"Processed vitals data: {vitals_data}")
            
            # Convert temperature and CO2 to float if provided
            if vitals_data['temperature'] is not None:
                vitals_data['temperature'] = float(vitals_data['temperature'])
            if vitals_data['co2_level'] is not None:
                vitals_data['co2_level'] = float(vitals_data['co2_level'])
            
            # Validate data
            if vitals_data['hr'] <= 0 or vitals_data['spo2'] <= 0:
                logger.warning(f"Invalid vitals data received: HR={vitals_data['hr']}, SpO2={vitals_data['spo2']}")
                logger.warning(f"Raw payload was: {data}")
                logger.warning(f"Payload data was: {payload_data}")
                logger.warning(f"Field mapping: hr from {payload_data.get('hr', 'NOT_FOUND')} or {payload_data.get('heartbeatBpm', 'NOT_FOUND')}")
                logger.warning(f"Field mapping: spo2 from {payload_data.get('spo2', 'NOT_FOUND')}")
                return
            
            # Log received data
            env_info = ""
            if vitals_data['temperature'] is not None:
                env_info += f", Temp={vitals_data['temperature']}°C"
            if vitals_data['co2_level'] is not None:
                env_info += f", CO2={vitals_data['co2_level']} PPM"
            
            logger.debug(f"Received vitals: HR={vitals_data['hr']} BPM, SpO2={vitals_data['spo2']}%{env_info}")
            
            # Call data callback if set
            if self.data_callback:
                try:
                    self.data_callback(vitals_data)
                except Exception as e:
                    logger.error(f"Error in data callback: {e}")
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON message: {e}")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid data format in message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def run_with_reconnect(self):
        """
        Run WebSocket client with automatic reconnection.
        """
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                await self.connect()
                await self.listen()
                
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self.is_connected = False
                
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    self.reconnect_attempts += 1
                    logger.info(f"Reconnecting in {self.reconnect_interval} seconds... (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
                    await asyncio.sleep(self.reconnect_interval)
                else:
                    logger.error("Max reconnection attempts reached. Stopping.")
                    break
    
    async def send_message(self, message: Dict[str, Any]):
        """
        Send message to WebSocket server.
        
        Args:
            message: Dictionary to send as JSON
        """
        if not self.is_connected or not self.websocket:
            raise RuntimeError("Not connected to WebSocket server")
        
        try:
            await self.websocket.send(json.dumps(message))
            logger.debug(f"Sent message: {message}")
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise


# Example usage
async def example_vitals_callback(vitals_data: Dict[str, Any]):
    """
    Example callback function for processing vitals data.
    
    Args:
        vitals_data: Dictionary containing timestamp, hr, spo2, quality, temperature, co2_level
                   (converted from WebSocket event type="inference.vital_info" with heartbeatBpm, spo2, temperatureC, co2Level, timestampMs)
    """
    timestamp = vitals_data['timestamp']
    hr = vitals_data['hr']
    spo2 = vitals_data['spo2']
    quality = vitals_data['quality']
    temperature = vitals_data.get('temperature')
    co2_level = vitals_data.get('co2_level')
    
    # Format timestamp for display
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    
    print(f"📊 Vitals Data:")
    print(f"   Time: {time_str}")
    print(f"   HR: {hr} BPM")
    print(f"   SpO2: {spo2}%")
    print(f"   Quality: {quality}")
    
    if temperature is not None:
        print(f"   Temperature: {temperature}°C")
    if co2_level is not None:
        print(f"   CO2 Level: {co2_level} PPM")
    
    print("-" * 40)


async def main():
    """Example usage of WebSocket client."""
    # Create client with authentication
    # Option 1: Use environment variables
    # export VITALS_API_KEY="your-api-key-here"
    # export VITALS_AUTH_TOKEN="your-bearer-token-here"
    client = VitalsWebSocketClient(uri="ws://localhost:8765")
    
    # Option 2: Set credentials directly
    # client = VitalsWebSocketClient(
    #     uri="ws://your-backend:8765",
    #     api_key="your-api-key-here",
    #     auth_token="your-bearer-token-here"
    # )
    
    # Option 3: Set custom headers
    # client = VitalsWebSocketClient(
    #     uri="ws://your-backend:8765",
    #     headers={
    #         "X-API-Key": "your-api-key",
    #         "Authorization": "Bearer your-token",
    #         "X-Client-ID": "vigilant-ai-client"
    #     }
    # )
    
    # Set callback for received data
    client.set_data_callback(example_vitals_callback)
    
    # Set connection callbacks
    client.set_connection_callback(lambda: print("🔗 Connected to server"))
    client.set_disconnection_callback(lambda: print("🔌 Disconnected from server"))
    
    try:
        # Run with automatic reconnection
        await client.run_with_reconnect()
    except KeyboardInterrupt:
        print("\n🛑 Stopping WebSocket client...")
        await client.disconnect()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the client
    asyncio.run(main())
"""
WebSocket Client for Receiving HR and SpO2 Data
Connects to backend WebSocket server to receive real-time vitals data.
"""

import asyncio
import json
import logging
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
                 reconnect_interval: int = 5,
                 max_reconnect_attempts: int = 10):
        """
        Initialize WebSocket client.
        
        Args:
            uri: WebSocket server URI
            reconnect_interval: Seconds to wait before reconnecting
            max_reconnect_attempts: Maximum number of reconnection attempts
        """
        self.uri = uri
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.websocket = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.data_callback: Optional[Callable] = None
        self.connection_callback: Optional[Callable] = None
        self.disconnection_callback: Optional[Callable] = None
        
    def set_data_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Set callback function to handle received vitals data.
        
        Args:
            callback: Function that receives vitals data dict with keys:
                     - timestamp: Unix timestamp
                     - hr: Heart rate in BPM
                     - spo2: Blood oxygen saturation percentage
                     - quality: Data quality indicator (optional)
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
        """Connect to WebSocket server."""
        try:
            logger.info(f"Connecting to WebSocket server: {self.uri}")
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
            message: JSON string containing vitals data
        """
        try:
            # Parse JSON message
            data = json.loads(message)
            
            # Extract vitals data
            vitals_data = {
                'timestamp': data.get('timestamp', time.time()),
                'hr': float(data.get('hr', 0)),
                'spo2': float(data.get('spo2', 0)),
                'quality': data.get('quality', 'unknown')
            }
            
            # Validate data
            if vitals_data['hr'] <= 0 or vitals_data['spo2'] <= 0:
                logger.warning(f"Invalid vitals data received: HR={vitals_data['hr']}, SpO2={vitals_data['spo2']}")
                return
            
            # Log received data
            logger.debug(f"Received vitals: HR={vitals_data['hr']} BPM, SpO2={vitals_data['spo2']}%")
            
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
        vitals_data: Dictionary containing timestamp, hr, spo2, quality
    """
    timestamp = vitals_data['timestamp']
    hr = vitals_data['hr']
    spo2 = vitals_data['spo2']
    quality = vitals_data['quality']
    
    # Format timestamp for display
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
    
    print(f"📊 Vitals Data:")
    print(f"   Time: {time_str}")
    print(f"   HR: {hr} BPM")
    print(f"   SpO2: {spo2}%")
    print(f"   Quality: {quality}")
    print("-" * 40)


async def main():
    """Example usage of WebSocket client."""
    # Create client
    client = VitalsWebSocketClient(uri="ws://localhost:8765")
    
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
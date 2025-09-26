"""
WebSocket Event Sender Module
Simple module to send events to WebSocket server at ws://192.168.5.93:4000/ws?token=dev-shared-secret
"""

import asyncio
import json
import logging
import time
import threading
from typing import Dict, Any, Optional, Union
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)


class WebSocketEventSender:
    """
    Simple WebSocket client for sending events to the server.
    """
    
    def __init__(self, uri: str = "ws://192.168.5.93:4000/ws?token=dev-shared-secret"):
        """
        Initialize WebSocket event sender.
        
        Args:
            uri: WebSocket server URI
        """
        self.uri = uri
        self.websocket = None
        self.is_connected = False
        
    async def connect(self) -> bool:
        """
        Connect to WebSocket server.
        
        Returns:
            bool: True if connected successfully, False otherwise
        """
        try:
            logger.info(f"Connecting to WebSocket server: {self.uri}")
            self.websocket = await websockets.connect(self.uri)
            self.is_connected = True
            logger.info("✅ Connected to WebSocket server")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to WebSocket server: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.websocket:
            try:
                await self.websocket.close()
                self.is_connected = False
                logger.info("🔌 Disconnected from WebSocket server")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
    
    async def send_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """
        Send an event to the WebSocket server.
        Uses a fresh connection for each message since server closes connection after each message.
        
        Args:
            event_type: Type of event (e.g., "health_score", "alert", "status")
            data: Event data dictionary
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        try:
            # Use a fresh connection for each message since server closes after each send
            logger.info(f"Connecting to send {event_type} event...")
            print(1111111)
            async with websockets.connect(self.uri) as websocket:
                # Send the message
                print(2222222)
                message_str = json.dumps(data)
                print(3333333)
                await websocket.send(message_str)
                logger.info(f"📤 Sent {event_type} event successfully")
                print(4444444)
                # Wait a moment for server to process
                await asyncio.sleep(0.1)
                print(5555555)
                return True
            
        except ConnectionClosed as e:
            logger.info(f"✅ Connection closed after sending {event_type} (this is normal): {e}")
            return True  # This is actually success since server closes after processing
            
        except Exception as e:
            logger.error(f"❌ Failed to send {event_type} event: {e}")
            return False
    
    async def send_health_score(self, health_score: float, dv_score: float = None, 
                               hv_score: float = None, **kwargs) -> bool:
        """
        Send health score event.
        
        Args:
            health_score: Health score value (0-1)
            dv_score: Drowsiness/Vision score (optional)
            hv_score: Health/Vitals score (optional)
            **kwargs: Additional data
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        data = {
            "type": "health_score",
            "userId": "dcc7fb95-2751-4149-8058-5a84a06b68b5",
            "payload": {
                "score": health_score,
                "reasons": ["well rested", "good heart rate", "low stress level"]
            }
        }
        
        # if dv_score is not None:
        #     data["dv_score"] = round(dv_score, 3)
        # if hv_score is not None:
        #     data["hv_score"] = round(hv_score, 3)
        
        # # Add interpretation
        # if health_score >= 0.8:
        #     data["interpretation"] = "Excellent Health"
        #     data["risk_level"] = "very_low"
        # elif health_score >= 0.6:
        #     data["interpretation"] = "Good Health"
        #     data["risk_level"] = "low"
        # elif health_score >= 0.4:
        #     data["interpretation"] = "Fair Health"
        #     data["risk_level"] = "moderate"
        # else:
        #     data["interpretation"] = "Poor Health"
        #     data["risk_level"] = "high"
        
        # Add any additional data
        # data.update(kwargs)
        print("DATA:################# ", data)
        
        return await self.send_event("health_score", data)


# Global instance for easy access
_event_sender = None
_sender_lock = threading.Lock()


def get_event_sender() -> WebSocketEventSender:
    """Get or create global WebSocket event sender instance."""
    global _event_sender
    with _sender_lock:
        if _event_sender is None:
            _event_sender = WebSocketEventSender()
        return _event_sender


async def send_health_score_event(health_score: float, dv_score: float = None, 
                                 hv_score: float = None, **kwargs) -> bool:
    """
    Convenience function to send health score event.
    
    Args:
        health_score: Health score value (0-1)
        dv_score: Drowsiness/Vision score (optional)
        hv_score: Health/Vitals score (optional)
        **kwargs: Additional data
        
    Returns:
        bool: True if sent successfully, False otherwise
    """
    sender = get_event_sender()
    
    # Connect if not connected
    if not sender.is_connected:
        if not await sender.connect():
            return False
    
    return await sender.send_health_score(health_score, dv_score, hv_score, **kwargs)


def send_health_score_sync(health_score: float, dv_score: float = None, 
                          hv_score: float = None, **kwargs) -> bool:
    """
    Synchronous wrapper to send health score event.
    Works from any context (sync/async/threaded).
    
    Args:
        health_score: Health score value (0-1)
        dv_score: Drowsiness/Vision score (optional)
        hv_score: Health/Vitals score (optional)
        **kwargs: Additional data
        
    Returns:
        bool: True if sent successfully, False otherwise
    """
    def run_in_thread():
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    send_health_score_event(health_score, dv_score, hv_score, **kwargs)
                )
                if result:
                    logger.info(f"📤 Health score sent successfully: {health_score:.3f}")
                else:
                    logger.warning(f"⚠️  Failed to send health score: {health_score:.3f}")
                return result
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"❌ Error sending health score: {e}")
            return False
    
    try:
        # Always run in a separate thread to avoid blocking and event loop conflicts
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        
        # Return True immediately - actual sending happens in background
        # This prevents blocking the main inference engine
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in send_health_score_sync: {e}")
        return False


async def send_event(event_type: str, data: Dict[str, Any]) -> bool:
    """
    Convenience function to send any event.
    
    Args:
        event_type: Type of event
        data: Event data
        
    Returns:
        bool: True if sent successfully, False otherwise
    """
    sender = get_event_sender()
    
    # Connect if not connected
    if not sender.is_connected:
        if not await sender.connect():
            return False
    
    return await sender.send_event(event_type, data)


def send_event_sync(event_type: str, data: Dict[str, Any]) -> bool:
    """
    Synchronous wrapper to send any event.
    
    Args:
        event_type: Type of event
        data: Event data
        
    Returns:
        bool: True if sent successfully, False otherwise
    """
    try:
        # Try to get current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Event loop is running, schedule the task
            asyncio.create_task(send_event(event_type, data))
            return True
        else:
            # Event loop exists but not running
            return loop.run_until_complete(send_event(event_type, data))
    except RuntimeError:
        # No event loop in current thread, run in new thread
        def run_in_thread():
            try:
                return asyncio.run(send_event(event_type, data))
            except Exception as e:
                logger.error(f"❌ Error sending {event_type} event in thread: {e}")
                return False
        
        # Run in background thread
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        return True
    except Exception as e:
        logger.error(f"❌ Error in send_event_sync: {e}")
        return False


# Example usage and testing
async def test_event_sender():
    """Test the WebSocket event sender."""
    print("🧪 Testing WebSocket Event Sender")
    print("=" * 35)
    
    sender = WebSocketEventSender()
    
    # Test connection
    print("🔌 Connecting to WebSocket...")
    if await sender.connect():
        print("✅ Connected successfully!")
        
        # Test health score event
        print("\n📤 Sending health score event...")
        success = await sender.send_health_score(
            health_score=0.75,
            dv_score=0.6,
            hv_score=0.9,
            mode="synchronized",
            source="test_event_sender"
        )
        print(f"Health score sent: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        # Test custom event
        print("\n📤 Sending custom event...")
        success = await sender.send_event("test_event", {
            "message": "Hello from event sender!",
            "test_id": "event_sender_test",
            "data": {"value": 42, "status": "active"}
        })
        print(f"Custom event sent: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        # Test convenience function
        print("\n📤 Testing convenience function...")
        success = await send_health_score_event(0.68, dv_score=0.5, hv_score=0.8)
        print(f"Convenience function: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        await sender.disconnect()
    else:
        print("❌ Connection failed")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    asyncio.run(test_event_sender())

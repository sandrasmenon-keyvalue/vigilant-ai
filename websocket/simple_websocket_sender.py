"""
Simple WebSocket Sender
A simple, reliable WebSocket sender that avoids asyncio complications.
"""

import json
import logging
import time
import threading
import queue
from typing import Dict, Any, Optional
import websockets
import asyncio

logger = logging.getLogger(__name__)


class SimpleWebSocketSender:
    """
    Simple WebSocket sender that runs in its own thread to avoid event loop conflicts.
    """
    
    def __init__(self, uri: str = "ws://192.168.5.93:4000/ws?token=dev-shared-secret"):
        """
        Initialize the WebSocket sender.
        
        Args:
            uri: WebSocket server URI
        """
        self.uri = uri
        self.message_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.websocket = None
        self.connected = False
        
    def start(self):
        """Start the WebSocket sender thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_sender, daemon=True)
            self.thread.start()
            logger.info("🚀 WebSocket sender thread started")
    
    def stop(self):
        """Stop the WebSocket sender thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("🛑 WebSocket sender thread stopped")
    
    def send_health_score(self, health_score: float, **kwargs) -> bool:
        """
        Send health score message.
        
        Args:
            health_score: Health score value (0-1)
            **kwargs: Additional data
            
        Returns:
            bool: True if queued successfully, False otherwise
        """
        try:
            message = {
                "type": "health_score",
                "userId": "dcc7fb95-2751-4149-8058-5a84a06b68b5",
                "payload": {
                    "score": health_score,
                    "reasons": ["well rested", "good heart rate", "low stress level"]
                }
                }
            
            # Start sender if not running
            if not self.running:
                self.start()
            
            # Queue the message
            self.message_queue.put(message, timeout=1)
            logger.debug(f"📝 Queued health score: {health_score:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to queue health score: {e}")
            return False
    
    def send_alert(self, alert_type: str, reason: str, alert_level: str = "warning", **kwargs) -> bool:
        """
        Send alert message.
        
        Args:
            alert_type: Type of alert (dv_high, hv_low, hr_violations, etc.)
            reason: Alert reason/message
            alert_level: Alert level (warning, critical, info)
            **kwargs: Additional data
            
        Returns:
            bool: True if queued successfully, False otherwise
        """
        try:
            message = {
                "type": "speech_alert",
                "userId": "dcc7fb95-2751-4149-8058-5a84a06b68b5",
                "payload": {
                    "text": reason,
                    "description":""
                }
            }
            
            # Start sender if not running
            if not self.running:
                self.start()
            
            # Queue the message
            self.message_queue.put(message, timeout=1)
            logger.debug(f"📝 Queued alert: {alert_type} - {reason}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to queue alert: {e}")
            return False
    
    def send_message(self, message: Dict[str, Any]) -> bool:
        """
        Send any message.
        
        Args:
            message: Message dictionary
            
        Returns:
            bool: True if queued successfully, False otherwise
        """
        try:
            # Start sender if not running
            if not self.running:
                self.start()
            
            # Queue the message
            self.message_queue.put(message, timeout=1)
            logger.debug(f"📝 Queued message: {message.get('type', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to queue message: {e}")
            return False
    
    def _run_sender(self):
        """Run the WebSocket sender in its own thread with its own event loop."""
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._sender_loop())
        except Exception as e:
            logger.error(f"❌ Error in sender loop: {e}")
        finally:
            loop.close()
    
    async def _sender_loop(self):
        """Main sender loop that handles WebSocket connection and message sending."""
        reconnect_delay = 1
        max_reconnect_delay = 30
        
        while self.running:
            try:
                # Connect to WebSocket
                logger.info(f"Connecting to WebSocket: {self.uri}")
                async with websockets.connect(self.uri) as websocket:
                    self.websocket = websocket
                    self.connected = True
                    reconnect_delay = 1  # Reset delay on successful connection
                    logger.info("✅ WebSocket connected")
                    
                    # Process messages
                    while self.running and self.connected:
                        try:
                            # Check for messages to send (non-blocking)
                            try:
                                message = self.message_queue.get_nowait()
                                
                                # Send the message
                                message_str = json.dumps(message)
                                await websocket.send(message_str)
                                logger.info(f"📤 Sent {message.get('type', 'message')}: {message.get('payload', 'N/A')}")
                                
                                # Mark task as done
                                self.message_queue.task_done()
                                
                            except queue.Empty:
                                # No messages to send, wait a bit
                                await asyncio.sleep(0.1)
                                
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("⚠️  WebSocket connection closed by server")
                            self.connected = False
                            break
                        except Exception as e:
                            logger.error(f"❌ Error sending message: {e}")
                            await asyncio.sleep(0.5)
                            
            except Exception as e:
                logger.error(f"❌ WebSocket connection error: {e}")
                self.connected = False
                
                if self.running:
                    logger.info(f"🔄 Reconnecting in {reconnect_delay} seconds...")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)


# Global sender instance
_sender = None
_sender_lock = threading.Lock()


def get_sender() -> SimpleWebSocketSender:
    """Get or create global WebSocket sender instance."""
    global _sender
    with _sender_lock:
        if _sender is None:
            _sender = SimpleWebSocketSender()
        return _sender


def send_health_score_simple(health_score: float, **kwargs) -> bool:
    """
    Send health score using the simple WebSocket sender.
    
    Args:
        health_score: Health score value (0-1)
        **kwargs: Additional data
        
    Returns:
        bool: True if queued successfully, False otherwise
    """
    sender = get_sender()
    return sender.send_health_score(health_score, **kwargs)


def send_alert_simple(alert_type: str, reason: str, alert_level: str = "warning", **kwargs) -> bool:
    """
    Send alert using the simple WebSocket sender.
    
    Args:
        alert_type: Type of alert (dv_high, hv_low, hr_violations, etc.)
        reason: Alert reason/message
        alert_level: Alert level (warning, critical, info)
        **kwargs: Additional data
        
    Returns:
        bool: True if queued successfully, False otherwise
    """
    sender = get_sender()
    return sender.send_alert(alert_type, reason, alert_level, **kwargs)


def send_message_simple(message: Dict[str, Any]) -> bool:
    """
    Send message using the simple WebSocket sender.
    
    Args:
        message: Message dictionary
        
    Returns:
        bool: True if queued successfully, False otherwise
    """
    sender = get_sender()
    return sender.send_message(message)


# Test function
def test_simple_sender():
    """Test the simple WebSocket sender."""
    print("🧪 Testing Simple WebSocket Sender")
    print("=" * 35)
    
    # Test health score sending
    print("📤 Sending health score...")
    success = send_health_score_simple(0.75, dv_score=0.6, hv_score=0.9)
    print(f"Health score queued: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    # Test alert sending
    print("📤 Sending alert...")
    success = send_alert_simple("dv_high", "Driver appears drowsy. Please take a break.", "warning")
    print(f"Alert queued: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    # Test critical alert
    print("📤 Sending critical alert...")
    success = send_alert_simple("hr_violations", "Heart rate is abnormal. Please check your condition.", "critical", hr_value=120)
    print(f"Critical alert queued: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    # Test custom message
    print("📤 Sending custom message...")
    success = send_message_simple({
        "type": "test",
        "message": "Hello from simple sender!",
        "timestamp": time.time()
    })
    print(f"Custom message queued: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    # Wait for messages to be sent
    print("⏳ Waiting 3 seconds for messages to be sent...")
    time.sleep(3)
    
    print("✅ Test completed!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    test_simple_sender()
    
    # Keep running for a bit to see the results
    time.sleep(5)

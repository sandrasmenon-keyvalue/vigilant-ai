"""
Vitals Processing package for Vigilant AI
Contains vitals processing and WebSocket integration.
"""

from .vitals_processor import VitalsProcessor
from .vitals_websocket_processor import VitalsWebSocketProcessor

__all__ = ['VitalsProcessor', 'VitalsWebSocketProcessor']

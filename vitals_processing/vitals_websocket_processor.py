"""
Integrated Vitals WebSocket Processor
Combines WebSocket client with vitals processing for real-time HR and SpO2 analysis.
"""

import asyncio
import logging
import time
import sys
import os
from typing import Dict, Any, Optional, Callable

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vitals_processing.vitals_processor import VitalsProcessor
from websocket.websocket_client import VitalsWebSocketClient

logger = logging.getLogger(__name__)


class VitalsWebSocketProcessor:
    """
    Integrated processor that receives HR and SpO2 data via WebSocket
    and processes it using the trained vitals model.
    """
    
    def __init__(self, 
                 websocket_uri: str = "ws://localhost:8765",
                 api_key: Optional[str] = None,
                 auth_token: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None,
                 age: int = 35,
                 health_conditions: Optional[Dict[str, int]] = None,
                 model_path: str = "trained_models/vital-training/vitals_hv_model_xgboost.pkl",
                 scaler_path: str = "trained_models/vital-training/vitals_feature_scaler_xgboost.pkl"):
        """
        Initialize the integrated vitals processor.
        
        Args:
            websocket_uri: WebSocket server URI
            api_key: API key for authentication (can also be set via VITALS_API_KEY env var)
            auth_token: Bearer token for authentication (can also be set via VITALS_AUTH_TOKEN env var)
            headers: Additional headers to send with connection
            age: Patient age for vitals processing
            health_conditions: Health conditions dict
            model_path: Path to trained vitals model
            scaler_path: Path to feature scaler
        """
        self.age = age
        self.health_conditions = health_conditions or {}
        
        # Initialize vitals processor
        self.vitals_processor = VitalsProcessor(model_path, scaler_path)
        
        # Initialize WebSocket client with authentication
        self.websocket_client = VitalsWebSocketClient(
            uri=websocket_uri,
            api_key=api_key,
            auth_token=auth_token,
            headers=headers
        )
        
        # Set up callbacks
        self.websocket_client.set_data_callback(self._process_vitals_data)
        self.websocket_client.set_connection_callback(self._on_connection)
        self.websocket_client.set_disconnection_callback(self._on_disconnection)
        
        # External callbacks
        self.hv_prediction_callback: Optional[Callable] = None
        self.raw_vitals_callback: Optional[Callable] = None
        self.connection_status_callback: Optional[Callable] = None
        
        # Statistics
        self.stats = {
            'total_messages': 0,
            'processed_predictions': 0,
            'connection_time': None,
            'last_data_time': None
        }
    
    def set_hv_prediction_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Set callback for HV prediction results.
        
        Args:
            callback: Function that receives HV prediction results
        """
        self.hv_prediction_callback = callback
    
    def set_raw_vitals_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Set callback for raw vitals data.
        
        Args:
            callback: Function that receives raw vitals data
        """
        self.raw_vitals_callback = callback
    
    def set_connection_status_callback(self, callback: Callable[[bool], None]):
        """
        Set callback for connection status changes.
        
        Args:
            callback: Function that receives connection status (True/False)
        """
        self.connection_status_callback = callback
    
    def set_health_conditions(self, health_conditions: Dict[str, int]):
        """
        Update health conditions for the patient.
        
        Args:
            health_conditions: Dictionary with health condition flags
        """
        self.health_conditions = health_conditions
        self.vitals_processor.set_health_conditions(health_conditions)
        logger.info(f"Health conditions updated: {health_conditions}")
    
    def set_age(self, age: int):
        """
        Update patient age.
        
        Args:
            age: Age in years
        """
        self.age = age
        logger.info(f"Patient age updated: {age}")
    
    def _process_vitals_data(self, vitals_data: Dict[str, Any]):
        """
        Process incoming vitals data from WebSocket.
        
        Args:
            vitals_data: Dictionary with timestamp, hr, spo2, quality, temperature, co2_level
                       (converted from WebSocket payload with heartbeatBpm, spo2, temperatureC, co2Level, timestampMs)
        """
        self.stats['total_messages'] += 1
        self.stats['last_data_time'] = time.time()
        
        # Call raw vitals callback if set
        if self.raw_vitals_callback:
            try:
                self.raw_vitals_callback(vitals_data)
            except Exception as e:
                logger.error(f"Error in raw vitals callback: {e}")
        
        # Process vitals for HV prediction
        try:
            hv_result = self.vitals_processor.process_websocket_vitals(
                vitals_data, 
                age=self.age, 
                health_conditions=self.health_conditions
            )
            
            # Add original vitals data to result
            hv_result['original_vitals'] = vitals_data
            
            self.stats['processed_predictions'] += 1
            
            # Call HV prediction callback if set
            if self.hv_prediction_callback:
                try:
                    self.hv_prediction_callback(hv_result)
                except Exception as e:
                    logger.error(f"Error in HV prediction callback: {e}")
            
            # Log prediction
            logger.info(f"HV Prediction: {hv_result['hv_score']:.3f} ({hv_result['risk_level']} risk)")
            
        except Exception as e:
            logger.error(f"Error processing vitals for HV prediction: {e}")
    
    def _on_connection(self):
        """Handle successful WebSocket connection."""
        self.stats['connection_time'] = time.time()
        logger.info("🔗 Connected to vitals WebSocket server")
        
        if self.connection_status_callback:
            try:
                self.connection_status_callback(True)
            except Exception as e:
                logger.error(f"Error in connection status callback: {e}")
    
    def _on_disconnection(self):
        """Handle WebSocket disconnection."""
        logger.warning("🔌 Disconnected from vitals WebSocket server")
        
        if self.connection_status_callback:
            try:
                self.connection_status_callback(False)
            except Exception as e:
                logger.error(f"Error in connection status callback: {e}")
    
    async def start(self):
        """
        Start the WebSocket processor with automatic reconnection.
        """
        logger.info("Starting Vitals WebSocket Processor...")
        logger.info(f"WebSocket URI: {self.websocket_client.uri}")
        logger.info(f"Patient Age: {self.age}")
        logger.info(f"Health Conditions: {self.health_conditions}")
        
        try:
            await self.websocket_client.run_with_reconnect()
        except KeyboardInterrupt:
            logger.info("🛑 Stopping Vitals WebSocket Processor...")
        except Exception as e:
            logger.error(f"Unexpected error in processor: {e}")
        finally:
            await self.websocket_client.disconnect()
    
    async def stop(self):
        """Stop the WebSocket processor."""
        await self.websocket_client.disconnect()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        stats = self.stats.copy()
        
        # Add connection duration if connected
        if self.websocket_client.is_connected and stats['connection_time']:
            stats['connection_duration'] = time.time() - stats['connection_time']
        
        # Add time since last data
        if stats['last_data_time']:
            stats['time_since_last_data'] = time.time() - stats['last_data_time']
        
        return stats
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.websocket_client.is_connected


# Example usage and callbacks
def example_hv_prediction_callback(hv_result: Dict[str, Any]):
    """
    Example callback for HV prediction results.
    
    Args:
        hv_result: Dictionary containing HV prediction and original vitals
    """
    vitals = hv_result['original_vitals']
    hv_score = hv_result['hv_score']
    risk_level = hv_result['risk_level']
    
    print(f"🏥 HV Analysis:")
    print(f"   HR: {vitals['hr']} BPM")
    print(f"   SpO2: {vitals['spo2']}%")
    print(f"   HV Score: {hv_score:.3f}")
    print(f"   Risk Level: {risk_level}")
    print(f"   Interpretation: {hv_result['interpretation']}")
    
    # Display environmental conditions if available
    if hv_result.get('environmental_conditions'):
        env = hv_result['environmental_conditions']
        print(f"🌡️ Environmental Conditions:")
        print(f"   Temperature: {env['temperature']['value']}°C ({env['temperature']['status']})")
        print(f"   CO2 Level: {env['co2_level']['value']} PPM ({env['co2_level']['status']})")
        print(f"   Overall Risk: {env['overall_risk']}")
        
        if env['alerts']:
            print(f"⚠️ Environmental Alerts:")
            for alert in env['alerts']:
                print(f"   - {alert}")
    
    print("-" * 50)


def example_raw_vitals_callback(vitals_data: Dict[str, Any]):
    """
    Example callback for raw vitals data.
    
    Args:
        vitals_data: Dictionary containing timestamp, hr, spo2, quality, temperature, co2_level
                   (converted from WebSocket payload with heartbeatBpm, spo2, temperatureC, co2Level, timestampMs)
    """
    timestamp = vitals_data['timestamp']
    hr = vitals_data['hr']
    spo2 = vitals_data['spo2']
    quality = vitals_data['quality']
    temperature = vitals_data.get('temperature')
    co2_level = vitals_data.get('co2_level')
    
    time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
    env_info = ""
    if temperature is not None:
        env_info += f", Temp={temperature}°C"
    if co2_level is not None:
        env_info += f", CO2={co2_level} PPM"
    
    print(f"📊 Raw Vitals [{time_str}]: HR={hr} BPM, SpO2={spo2}%, Quality={quality}{env_info}")


def example_connection_status_callback(is_connected: bool):
    """
    Example callback for connection status changes.
    
    Args:
        is_connected: True if connected, False if disconnected
    """
    status = "🔗 Connected" if is_connected else "🔌 Disconnected"
    print(f"WebSocket Status: {status}")


async def main():
    """Example usage of the integrated vitals processor."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up health conditions for the patient
    health_conditions = {
        'diabetes': 0,
        'hypertension': 1,  # Patient has hypertension
        'heart_disease': 0,
        'respiratory_condition': 0,
        'smoker': 0
    }
    
    # Create processor
    processor = VitalsWebSocketProcessor(
        websocket_uri="ws://localhost:8765",
        age=45,
        health_conditions=health_conditions
    )
    
    # Set up callbacks
    processor.set_hv_prediction_callback(example_hv_prediction_callback)
    processor.set_raw_vitals_callback(example_raw_vitals_callback)
    processor.set_connection_status_callback(example_connection_status_callback)
    
    try:
        # Start processing
        await processor.start()
    except KeyboardInterrupt:
        print("\n🛑 Stopping processor...")
        await processor.stop()
        
        # Print final statistics
        stats = processor.get_stats()
        print(f"\n📈 Final Statistics:")
        print(f"   Total Messages: {stats['total_messages']}")
        print(f"   Processed Predictions: {stats['processed_predictions']}")
        if 'connection_duration' in stats:
            print(f"   Connection Duration: {stats['connection_duration']:.1f} seconds")


if __name__ == "__main__":
    asyncio.run(main())

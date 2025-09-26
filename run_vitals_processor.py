#!/usr/bin/env python3
"""
Run Vitals WebSocket Processor
Standalone script to run the vitals WebSocket processor for HR and SpO2 analysis.
"""

import asyncio
import logging
import os
import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from vitals_processing.vitals_websocket_processor import VitalsWebSocketProcessor

# Import synchronized inference engine
try:
    from inference.synchronized_inference_engine import SynchronizedInferenceEngine
except ImportError:
    SynchronizedInferenceEngine = None
    print("⚠️  Warning: SynchronizedInferenceEngine not available")


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_hv_prediction_callback(hv_result: dict):
    """
    Callback for HV prediction results.
    
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


def example_raw_vitals_callback(vitals_data: dict):
    """
    Callback for raw vitals data.
    
    Args:
        vitals_data: Dictionary containing timestamp, hr, spo2, quality, temperature, co2_level
    """
    import time
    
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
    Callback for connection status changes.
    
    Args:
        is_connected: True if connected, False if disconnected
    """
    status = "🔗 Connected" if is_connected else "🔌 Disconnected"
    print(f"WebSocket Status: {status}")


async def main():
    """Main function to run the vitals processor."""
    parser = argparse.ArgumentParser(description='Run Vitals WebSocket Processor')
    parser.add_argument('--uri', default=None, 
                       help='WebSocket URI (default: from VITALS_WEBSOCKET_URI env var or ws://192.168.5.93:4000/ws?token=dev-shared-secret)')
    parser.add_argument('--age', type=int, default=35, 
                       help='Patient age (default: 35)')
    parser.add_argument('--api-key', default=None,
                       help='API key for authentication (default: from VITALS_API_KEY env var)')
    parser.add_argument('--auth-token', default=None,
                       help='Bearer token for authentication (default: from VITALS_AUTH_TOKEN env var)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--diabetes', action='store_true',
                       help='Patient has diabetes')
    parser.add_argument('--hypertension', action='store_true',
                       help='Patient has hypertension')
    parser.add_argument('--heart-disease', action='store_true',
                       help='Patient has heart disease')
    parser.add_argument('--respiratory-condition', action='store_true',
                       help='Patient has respiratory condition')
    parser.add_argument('--smoker', action='store_true',
                       help='Patient is a smoker')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Get WebSocket URI
    websocket_uri = args.uri or os.getenv('VITALS_WEBSOCKET_URI', 'ws://192.168.5.93:4000/ws?token=dev-shared-secret')
    
    # Get authentication credentials
    api_key = args.api_key or os.getenv('VITALS_API_KEY')
    auth_token = args.auth_token or os.getenv('VITALS_AUTH_TOKEN')
    
    # Set up health conditions
    health_conditions = {
        'diabetes': 1 if args.diabetes else 0,
        'hypertension': 1 if args.hypertension else 0,
        'heart_disease': 1 if args.heart_disease else 0,
        'respiratory_condition': 1 if args.respiratory_condition else 0,
        'smoker': 1 if args.smoker else 0
    }
    
    print("🚗 Vigilant AI - Vitals WebSocket Processor")
    print("=" * 50)
    print(f"WebSocket URI: {websocket_uri}")
    print(f"Patient Age: {args.age}")
    print(f"Health Conditions: {health_conditions}")
    print(f"API Key: {'✅ Set' if api_key else '❌ Not set'}")
    print(f"Auth Token: {'✅ Set' if auth_token else '❌ Not set'}")
    print("=" * 50)
    
    # Create processor (will connect to live stream service's sync engine via HTTP API)
    try:
        processor = VitalsWebSocketProcessor(
            websocket_uri=websocket_uri,
            api_key=api_key,
            auth_token=auth_token,
            age=args.age,
            health_conditions=health_conditions,
            live_stream_service_url="https://localhost:8001"  # Connect to live stream service
        )
        
        # Set up callbacks
        processor.set_hv_prediction_callback(example_hv_prediction_callback)
        processor.set_raw_vitals_callback(example_raw_vitals_callback)
        processor.set_connection_status_callback(example_connection_status_callback)
        
        print("🏥 Starting vitals processor...")
        print("🛑 Press Ctrl+C to stop")
        print()
        
        # Start processing
        await processor.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping processor...")
        if 'processor' in locals():
            await processor.stop()
        
        # Print final statistics
        if 'processor' in locals():
            stats = processor.get_stats()
            print(f"\n📈 Final Statistics:")
            print(f"   Total Messages: {stats['total_messages']}")
            print(f"   Processed Predictions: {stats['processed_predictions']}")
            if 'connection_duration' in stats:
                print(f"   Connection Duration: {stats['connection_duration']:.1f} seconds")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Start WebSocket Services for Vigilant AI
Starts both the live streaming websocket and vitals processing websocket services.
"""

import os
import sys
import time
import subprocess
import signal
import threading
import asyncio
import logging
from pathlib import Path
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def start_live_stream_service():
    """Start the live stream service with WebSocket endpoints."""
    print("🎥 Starting Live Stream Service (WebSocket) on port 8001...")
    try:
        process = subprocess.Popen([
            sys.executable, "live_stream_service.py"
        ], cwd=Path(__file__).parent)
        return process
    except Exception as e:
        print(f"❌ Failed to start live stream service: {e}")
        return None


def start_vitals_websocket_processor():
    """Start the vitals WebSocket processor."""
    print("🏥 Starting Vitals WebSocket Processor...")
    try:
        process = subprocess.Popen([
            sys.executable, "run_vitals_processor.py"
        ], cwd=Path(__file__).parent)
        return process
    except Exception as e:
        print(f"❌ Failed to start vitals processor: {e}")
        return None


def start_inference_api():
    """Start the inference API service (required for live stream)."""
    print("🧠 Starting Inference API Service on port 8002...")
    try:
        process = subprocess.Popen([
            sys.executable, "inference_api.py"
        ], cwd=Path(__file__).parent)
        return process
    except Exception as e:
        print(f"❌ Failed to start inference API: {e}")
        return None


def wait_for_service(url: str, service_name: str, timeout: int = 30):
    """Wait for a service to become available."""
    import requests
    
    print(f"⏳ Waiting for {service_name} to start...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Disable SSL verification for HTTPS endpoints (self-signed cert)
            verify_ssl = not url.startswith('https')
            response = requests.get(url, timeout=2, verify=verify_ssl)
            if response.status_code == 200:
                print(f"✅ {service_name} is ready!")
                return True
        except:
            pass
        
        time.sleep(1)
        print(".", end="", flush=True)
    
    print(f"\n❌ {service_name} did not start within {timeout} seconds")
    return False


def check_vitals_websocket_config():
    """Check if vitals WebSocket configuration is available."""
    vitals_uri = os.getenv('VITALS_WEBSOCKET_URI', 'ws://192.168.5.102:4000/ws?token=dev-shared-secret')
    api_key = os.getenv('VITALS_API_KEY')
    auth_token = os.getenv('VITALS_AUTH_TOKEN')
    
    print(f"🔧 Vitals WebSocket Configuration:")
    print(f"   URI: {vitals_uri}")
    print(f"   API Key: {'✅ Set' if api_key else '❌ Not set'}")
    print(f"   Auth Token: {'✅ Set' if auth_token else '❌ Not set'}")
    
    if not api_key and not auth_token:
        print("⚠️  Warning: No authentication credentials found.")
        print("   Set VITALS_API_KEY or VITALS_AUTH_TOKEN environment variables if your backend requires authentication.")
    
    return vitals_uri


def main():
    """Main function to start WebSocket services."""
    print("🚗 Vigilant AI - Starting WebSocket Services")
    print("=" * 60)
    
    # Check vitals configuration
    vitals_uri = check_vitals_websocket_config()
    
    processes = []
    
    try:
        # Start inference API first (required for live stream)
        inference_process = start_inference_api()
        if inference_process:
            processes.append(("Inference API", inference_process))
            
            # Wait for inference API to start
            if wait_for_service("http://localhost:8002/inference/health", "Inference API"):
                time.sleep(2)  # Give it a moment
                
                # Start live stream service (with WebSocket endpoints)
                stream_process = start_live_stream_service()
                if stream_process:
                    processes.append(("Live Stream Service", stream_process))
                    
                    # Wait for live stream service to start
                    if wait_for_service("https://localhost:8001/health", "Live Stream Service"):
                        
                        # Start vitals WebSocket processor (optional)
                        vitals_process = start_vitals_websocket_processor()
                        if vitals_process:
                            processes.append(("Vitals WebSocket Processor", vitals_process))
                            print("🏥 Vitals processor started (will attempt to connect to backend)")
                        else:
                            print("⚠️  Vitals processor not started (optional service)")
                        
                        print("\n🎉 Core WebSocket services started successfully!")
                        print("\n📡 WebSocket Endpoints:")
                        print("   Live Stream (Mobile):  wss://localhost:8001/ws/mobile_stream")
                        print("   Live Stream (Desktop): wss://localhost:8001/ws/live_stream")
                        print(f"   Vitals Backend:        {vitals_uri} (optional)")
                        print("\n📱 Client URLs:")
                        print("   Mobile Client:         https://localhost:8443/mobile_camera_client.html")
                        print("   Desktop Client:        https://localhost:8443/web_client.html")
                        print("\n📚 API Documentation:")
                        print("   Inference API:         http://localhost:8002/docs")
                        print("   Live Stream API:       https://localhost:8001/docs")
                        print("\n🔧 Vitals Configuration (Optional):")
                        print("   Set these environment variables if you have a vitals backend:")
                        print("     VITALS_WEBSOCKET_URI - Backend WebSocket URI")
                        print("     VITALS_API_KEY       - API key for vitals backend")
                        print("     VITALS_AUTH_TOKEN    - Bearer token for vitals backend")
                        print("\n💡 Note: Vitals processor will retry connection automatically")
                    else:
                        print("❌ Live stream service failed to start")
                else:
                    print("❌ Failed to start live stream service")
            else:
                print("❌ Inference API failed to start")
        else:
            print("❌ Failed to start inference API")
        
        if processes:
            print(f"\n🛑 Press Ctrl+C to stop all services")
            
            # Wait for interrupt
            try:
                # Keep main thread alive
                while True:
                    time.sleep(1)
                    
                    # Check if processes are still running
                    for name, process in processes:
                        if process.poll() is not None:
                            print(f"\n⚠️  {name} has stopped unexpectedly")
                            
            except KeyboardInterrupt:
                print("\n👋 Shutting down all services...")
    
    except Exception as e:
        print(f"❌ Error starting services: {e}")
    
    finally:
        # Cleanup all processes
        print("🛑 Stopping all services...")
        for name, process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                print(f"⏰ Force killing {name}...")
                process.kill()
                process.wait()
                print(f"✅ {name} force stopped")
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")


if __name__ == "__main__":
    main()

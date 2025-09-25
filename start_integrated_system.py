#!/usr/bin/env python3
"""
Start Integrated System
Starts all services required for the integrated system:
- Inference API
- Vitals WebSocket Server (simulated)
- Integrated Live Stream Service
"""

import os
import sys
import time
import subprocess
import signal
import threading
from pathlib import Path


def start_inference_api():
    """Start the inference API service."""
    print("🧠 Starting Inference API Service on port 8002...")
    try:
        process = subprocess.Popen([
            sys.executable, "inference_api.py"
        ], cwd=Path(__file__).parent)
        return process
    except Exception as e:
        print(f"❌ Failed to start inference API: {e}")
        return None


def start_vitals_websocket_server():
    """Start a simulated vitals WebSocket server for testing."""
    print("💓 Starting Vitals WebSocket Server on port 8765...")
    try:
        # Create a simple vitals WebSocket server
        vitals_server_code = '''
import asyncio
import json
import random
import time
import websockets
from datetime import datetime

async def vitals_websocket_server(websocket, path):
    """Simulated vitals WebSocket server."""
    print(f"🔗 Vitals WebSocket connected: {websocket.remote_address}")
    
    try:
        while True:
            # Generate simulated vitals data
            vitals_data = {
                "timestamp": time.time(),
                "hr": random.randint(60, 100),  # Heart rate 60-100 BPM
                "spo2": random.randint(95, 100),  # SpO2 95-100%
                "quality": "good",
                "temperature": random.uniform(18, 30),  # Temperature 18-30°C
                "co2_level": random.randint(400, 1500)  # CO2 400-1500 PPM
            }
            
            # Send vitals data
            await websocket.send(json.dumps(vitals_data))
            print(f"📊 Sent vitals: HR={vitals_data['hr']}, SpO2={vitals_data['spo2']}, "
                  f"Temp={vitals_data['temperature']:.1f}°C, CO2={vitals_data['co2_level']} PPM")
            
            # Wait 1 second before next data
            await asyncio.sleep(1)
            
    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 Vitals WebSocket disconnected: {websocket.remote_address}")
    except Exception as e:
        print(f"❌ Vitals WebSocket error: {e}")

async def main():
    print("💓 Starting Vitals WebSocket Server...")
    server = await websockets.serve(vitals_websocket_server, "localhost", 8765)
    print("✅ Vitals WebSocket Server running on ws://localhost:8765")
    
    # Keep server running
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        # Write the server code to a temporary file
        server_file = Path(__file__).parent / "temp_vitals_server.py"
        with open(server_file, 'w') as f:
            f.write(vitals_server_code)
        
        process = subprocess.Popen([
            sys.executable, str(server_file)
        ], cwd=Path(__file__).parent)
        
        return process, server_file
        
    except Exception as e:
        print(f"❌ Failed to start vitals WebSocket server: {e}")
        return None, None


def start_integrated_live_stream():
    """Start the integrated live stream service."""
    print("🎥 Starting Integrated Live Stream Service on port 8001...")
    try:
        process = subprocess.Popen([
            sys.executable, "integrated_live_stream_service.py"
        ], cwd=Path(__file__).parent)
        return process
    except Exception as e:
        print(f"❌ Failed to start integrated live stream service: {e}")
        return None


def wait_for_service(url: str, service_name: str, timeout: int = 30):
    """Wait for a service to become available."""
    import requests
    import urllib3
    
    # Disable SSL warnings for testing
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5, verify=False)
            if response.status_code == 200:
                print(f"✅ {service_name} is ready")
                return True
        except:
            pass
        time.sleep(1)
    
    print(f"❌ {service_name} did not start within {timeout} seconds")
    return False


def cleanup_temp_files(temp_files):
    """Clean up temporary files."""
    for temp_file in temp_files:
        try:
            if temp_file and temp_file.exists():
                temp_file.unlink()
                print(f"🧹 Cleaned up {temp_file}")
        except Exception as e:
            print(f"⚠️ Failed to clean up {temp_file}: {e}")


def main():
    """Main function to start all integrated services."""
    print("🚗 Vigilant AI - Starting Integrated System")
    print("=" * 60)
    print("This will start:")
    print("  • Inference API (port 8002)")
    print("  • Vitals WebSocket Server (port 8765)")
    print("  • Integrated Live Stream Service (port 8001)")
    print("=" * 60)
    
    processes = []
    temp_files = []
    
    try:
        # Start inference API first
        inference_process = start_inference_api()
        if inference_process:
            processes.append(("Inference API", inference_process))
            
            # Wait for inference API to start
            if wait_for_service("http://localhost:8002/inference/health", "Inference API"):
                time.sleep(2)  # Give it a moment
                
                # Start vitals WebSocket server
                vitals_process, temp_file = start_vitals_websocket_server()
                if vitals_process:
                    processes.append(("Vitals WebSocket Server", vitals_process))
                    if temp_file:
                        temp_files.append(temp_file)
                    
                    time.sleep(2)  # Give vitals server a moment
                    
                    # Start integrated live stream service
                    stream_process = start_integrated_live_stream()
                    if stream_process:
                        processes.append(("Integrated Live Stream Service", stream_process))
                        
                        # Wait for integrated service to start
                        if wait_for_service("http://localhost:8001/health", "Integrated Live Stream Service"):
                            
                            print("\n🎉 All integrated services started successfully!")
                            print("\n📡 Service URLs:")
                            print("   Inference API:     http://localhost:8002")
                            print("   Vitals WebSocket:  ws://localhost:8765")
                            print("   Integrated Stream: http://localhost:8001")
                            print("   WebSocket:         ws://localhost:8001/ws/integrated_stream")
                            
                            print("\n📚 Documentation:")
                            print("   Inference API:     http://localhost:8002/docs")
                            print("   Integrated API:    http://localhost:8001/docs")
                            
                            print("\n🧪 Testing:")
                            print("   Test Script:       python test_integrated_system.py")
                            
                            print("\n📋 Features:")
                            print("   ✅ DV data from inference_api.py")
                            print("   ✅ HV data from vitals WebSocket")
                            print("   ✅ Environmental monitoring (temperature, CO2)")
                            print("   ✅ Unified alert system")
                            print("   ✅ Health score calculation")
                            print("   ✅ Real-time processing")
                            
                            print("\n🛑 Press Ctrl+C to stop all services")
                            
                            # Keep running until interrupted
                            try:
                                while True:
                                    time.sleep(1)
                            except KeyboardInterrupt:
                                print("\n🛑 Stopping all services...")
                                
                        else:
                            print("❌ Integrated Live Stream Service failed to start")
                    else:
                        print("❌ Failed to start Integrated Live Stream Service")
                else:
                    print("❌ Failed to start Vitals WebSocket Server")
            else:
                print("❌ Inference API failed to start")
        else:
            print("❌ Failed to start Inference API")
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Stop all processes
        for service_name, process in processes:
            try:
                print(f"🛑 Stopping {service_name}...")
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {service_name} stopped")
            except Exception as e:
                print(f"⚠️ Error stopping {service_name}: {e}")
        
        # Clean up temporary files
        cleanup_temp_files(temp_files)
        
        print("✅ All services stopped")


if __name__ == "__main__":
    main()

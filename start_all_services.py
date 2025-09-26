#!/usr/bin/env python3
"""
Start All Vigilant AI Services
Starts both the inference API and live stream services.
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


def start_live_stream_service():
    """Start the live stream service."""
    print("🎥 Starting Live Stream Service on port 8001...")
    try:
        process = subprocess.Popen([
            sys.executable, "live_stream_service.py"
        ], cwd=Path(__file__).parent)
        return process
    except Exception as e:
        print(f"❌ Failed to start live stream service: {e}")
        return None


def start_https_server():
    """Start the HTTPS server for mobile client."""
    print("🔒 Starting HTTPS Server on port 8443...")
    try:
        process = subprocess.Popen([
            sys.executable, "https_server.py"
        ], cwd=Path(__file__).parent)
        return process
    except Exception as e:
        print(f"❌ Failed to start HTTPS server: {e}")
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


def main():
    """Main function to start all services."""
    print("🚗 Vigilant AI - Starting All Services")
    print("=" * 50)
    
    processes = []
    
    try:
        # Start inference API first
        inference_process = start_inference_api()
        if inference_process:
            processes.append(("Inference API", inference_process))
            
            # Wait for inference API to start
            if wait_for_service("http://localhost:8002/inference/health", "Inference API"):
                time.sleep(2)  # Give it a moment
                
                # Start live stream service
                stream_process = start_live_stream_service()
                if stream_process:
                    processes.append(("Live Stream Service", stream_process))
                    
                    # Wait for live stream service to start (now HTTPS)
                    if wait_for_service("https://localhost:8001/health", "Live Stream Service"):
                        
                        # Start HTTPS server for mobile client
                        https_process = start_https_server()
                        if https_process:
                            processes.append(("HTTPS Server", https_process))
                            
                            # Wait for HTTPS server to start
                            if wait_for_service("https://localhost:8443", "HTTPS Server (with -k flag)"):
                                
                                # Start vitals WebSocket processor (optional)
                                vitals_process = start_vitals_websocket_processor()
                                if vitals_process:
                                    processes.append(("Vitals WebSocket Processor", vitals_process))
                                    print("🏥 Vitals processor started (optional service)")
                                else:
                                    print("⚠️  Vitals processor not started (optional service)")
                                
                                print("\n🎉 All services started successfully!")
                                print("\n📡 Service URLs:")
                                print("   Inference API:     http://localhost:8002")
                                print("   Live Stream:       https://localhost:8001") 
                                print("   Mobile Client:     https://192.168.2.220:8443/mobile_camera_client.html")
                                print("   Desktop Client:    https://192.168.2.220:8443/web_client.html")
                                print("\n📚 Documentation:")
                                print("   Inference API:     http://localhost:8002/docs")
                                print("   Live Stream API:   https://localhost:8001/docs")
                                print("\n📱 Mobile Setup:")
                                print("   1. Open mobile browser: https://192.168.2.220:8443/mobile_camera_client.html")
                                print("   2. Accept security warning (self-signed cert)")
                                print("   3. Server URL: wss://192.168.2.220:8001 (Secure WebSocket)")
                                print("   4. Start Camera → Connect")
                                print("\n📋 WebSocket Endpoints:")
                                print("   Mobile Browser:    /ws/mobile_stream")
                                print("   Desktop/Local:     /ws/live_stream")
                                print("   Vitals Backend:    ws://192.168.5.93:4000/ws?token=dev-shared-secret (configurable)")
                                print("\n🔧 Vitals Configuration:")
                                print("   Set environment variables for vitals backend:")
                                print("     VITALS_WEBSOCKET_URI - Backend WebSocket URI")
                                print("     VITALS_API_KEY       - API key for authentication")
                                print("     VITALS_AUTH_TOKEN    - Bearer token for authentication")
                            else:
                                print("❌ HTTPS server failed to start")
                        else:
                            print("❌ Failed to start HTTPS server")
                        
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
                    
                    else:
                        print("❌ Live stream service failed to start")
                else:
                    print("❌ Failed to start live stream service")
            else:
                print("❌ Inference API failed to start")
        else:
            print("❌ Failed to start inference API")
    
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

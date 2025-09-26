#!/usr/bin/env python3
"""
Test script to verify drowsiness detection is working after the fix.
This script starts both services and provides instructions for testing.
"""

import subprocess
import time
import sys
import os
import signal
import requests
from pathlib import Path

def check_port(port):
    """Check if a port is in use."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result == 0

def start_service(script_name, port, service_name):
    """Start a service and return the process."""
    print(f"🚀 Starting {service_name}...")
    
    if check_port(port):
        print(f"⚠️  Port {port} is already in use. {service_name} might already be running.")
        return None
    
    try:
        process = subprocess.Popen([
            sys.executable, script_name
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Give the service time to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"✅ {service_name} started successfully on port {port}")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ {service_name} failed to start:")
            print(f"   stdout: {stdout}")
            print(f"   stderr: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting {service_name}: {e}")
        return None

def test_service_health(url, service_name):
    """Test if a service is healthy."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {service_name} health check passed")
            return True
        else:
            print(f"❌ {service_name} health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {service_name} health check error: {e}")
        return False

def main():
    print("🔧 Vigilant AI Drowsiness Detection - Test Fix")
    print("=" * 60)
    
    # Change to the project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    processes = []
    
    try:
        # Start Inference API (port 8002)
        inference_process = start_service("inference_api.py", 8002, "Inference API")
        if inference_process:
            processes.append(inference_process)
        
        # Wait a bit for inference API to fully start
        time.sleep(2)
        
        # Start Live Stream Service (port 8001)
        stream_process = start_service("live_stream_service.py", 8001, "Live Stream Service")
        if stream_process:
            processes.append(stream_process)
        
        # Wait for services to fully initialize
        time.sleep(5)
        
        print("\n🔍 Testing service health...")
        
        # Test Inference API health
        inference_healthy = test_service_health("http://localhost:8002/inference/health", "Inference API")
        
        # Test Live Stream Service health
        stream_healthy = test_service_health("http://localhost:8001/health", "Live Stream Service")
        
        if inference_healthy and stream_healthy:
            print("\n✅ All services are running successfully!")
            print("\n📱 Testing Instructions:")
            print("1. Open your mobile browser")
            print("2. Navigate to: https://localhost:8001/mobile_camera_client.html")
            print("   (Or use the IP address of this machine)")
            print("3. Allow camera permissions")
            print("4. Click 'Start Camera'")
            print("5. Click 'Connect Server'")
            print("6. You should now see drowsiness scores updating in real-time!")
            print("\n🔧 Fix Applied:")
            print("- Fixed batch processing logic to always return drowsiness scores")
            print("- Added better error handling for inference API failures")
            print("- Improved WebSocket communication reliability")
            print("\n💡 Expected Behavior:")
            print("- Drowsiness score should update every frame (not just every 2 frames)")
            print("- Score should start at 0.000 and change based on face detection")
            print("- Alert level should change colors based on score thresholds")
            print("- No more 'stuck at 0.000' issue")
            
            print(f"\n⏳ Services will run until you press Ctrl+C...")
            
            # Keep services running
            while True:
                time.sleep(1)
                
                # Check if processes are still alive
                for i, process in enumerate(processes):
                    if process.poll() is not None:
                        print(f"⚠️  Process {i} has terminated")
                        
        else:
            print("\n❌ Some services failed to start properly")
            print("Please check the error messages above and try again")
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        
    finally:
        # Clean up processes
        for process in processes:
            if process and process.poll() is None:
                print(f"🛑 Terminating process {process.pid}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"⚠️  Force killing process {process.pid}...")
                    process.kill()
        
        print("✅ All services stopped")

if __name__ == "__main__":
    main()

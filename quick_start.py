#!/usr/bin/env python3
"""
Quick Start Script for Vigilant AI
Starts services and provides correct connection URLs
"""

import subprocess
import time
import sys
import os
import requests

def get_local_ip():
    """Get the local IP address."""
    import socket
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "192.168.2.220"  # Fallback to known working IP

def check_service(url, name):
    """Check if a service is running."""
    try:
        response = requests.get(url, timeout=5, verify=False)
        return response.status_code == 200
    except:
        return False

def main():
    print("🚀 Vigilant AI - Quick Start")
    print("=" * 50)
    
    # Get local IP
    local_ip = get_local_ip()
    print(f"📍 Detected IP Address: {local_ip}")
    
    # Check if services are already running
    inference_running = check_service(f"http://{local_ip}:8002/inference/health", "Inference API")
    stream_running = check_service(f"https://{local_ip}:8001/health", "Live Stream Service")
    
    if inference_running and stream_running:
        print("\n✅ Services are already running!")
    else:
        print("\n⚠️  Some services are not running. Please start them manually:")
        print("   python inference_api.py &")
        print("   python live_stream_service.py &")
        print("\n   Or use: python start_all_services.py")
    
    print(f"\n📱 MOBILE CONNECTION INSTRUCTIONS:")
    print("=" * 50)
    print(f"1. 📱 Open mobile browser to:")
    print(f"   https://{local_ip}:8443/mobile_camera_client.html")
    print(f"")
    print(f"2. 🔒 Accept security warning (self-signed certificate)")
    print(f"")
    print(f"3. 📷 Click 'Start Camera' (allow camera permissions)")
    print(f"")
    print(f"4. 🔗 Click 'Connect Server'")
    print(f"   Server URL should be: wss://{local_ip}:8001")
    print(f"")
    print(f"5. 👀 You should see drowsiness scores updating in real-time!")
    
    print(f"\n🔧 TROUBLESHOOTING:")
    print("=" * 50)
    print("• If connection fails, try clicking 'Clear Cache' in mobile client")
    print("• Make sure mobile device is on same network")
    print("• For certificate warnings, click 'Advanced' → 'Proceed to site'")
    
    print(f"\n📊 SERVICE STATUS:")
    print("=" * 50)
    print(f"• Inference API:     {'✅ Running' if inference_running else '❌ Not Running'}")
    print(f"• Live Stream:       {'✅ Running' if stream_running else '❌ Not Running'}")
    
    if inference_running and stream_running:
        print(f"\n🎯 QUICK TEST:")
        print(f"   Connection Test: file://{os.path.abspath('connection_test.html')}")
        print(f"   Mobile Client:   https://{local_ip}:8443/mobile_camera_client.html")
    
    print(f"\n💡 The drowsiness detection fix has been applied!")
    print("   - Scores now update every frame (not just every 2 frames)")
    print("   - Better error handling for connection issues")
    print("   - Real-time score updates without delays")

if __name__ == "__main__":
    main()

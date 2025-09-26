#!/usr/bin/env python3
"""
Diagnose Vigilant AI Services
Quick diagnostic script to check service health and troubleshoot startup issues.
"""

import asyncio
import aiohttp
import requests
import time
import os
from pathlib import Path


async def check_service_async(url: str, service_name: str, use_ssl: bool = True):
    """Check service health asynchronously."""
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        connector = aiohttp.TCPConnector(ssl=False) if not use_ssl else None
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ {service_name}: {data.get('status', 'OK')}")
                    return True, data
                else:
                    print(f"❌ {service_name}: HTTP {response.status}")
                    return False, None
    except Exception as e:
        print(f"❌ {service_name}: {str(e)}")
        return False, None


def check_service_sync(url: str, service_name: str, verify_ssl: bool = False):
    """Check service health synchronously."""
    try:
        response = requests.get(url, timeout=5, verify=verify_ssl)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {service_name}: {data.get('status', 'OK')}")
            return True, data
        else:
            print(f"❌ {service_name}: HTTP {response.status_code}")
            return False, None
    except Exception as e:
        print(f"❌ {service_name}: {str(e)}")
        return False, None


def check_files():
    """Check required files."""
    print("\n🔍 Checking Required Files:")
    
    files_to_check = [
        "server.crt",
        "server.key", 
        "trained_models/vision-training/drowsiness_model.pkl",
        "trained_models/vision-training/feature_scaler.pkl",
        "trained_models/vital-training/vitals_hv_model_xgboost.pkl",
        "trained_models/vital-training/vitals_feature_scaler_xgboost.pkl",
        "models/face_landmarker.task"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} (missing)")


def check_ports():
    """Check if ports are available."""
    print("\n🔌 Checking Port Availability:")
    
    import socket
    
    ports_to_check = [
        (8001, "Live Stream Service"),
        (8002, "Inference API"),
        (8443, "HTTPS Server"),
        (8765, "Vitals WebSocket (default)")
    ]
    
    for port, service in ports_to_check:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"🟡 Port {port} ({service}): In use")
        else:
            print(f"✅ Port {port} ({service}): Available")


async def main():
    """Main diagnostic function."""
    print("🚗 Vigilant AI - Service Diagnostics")
    print("=" * 50)
    
    # Check files
    check_files()
    
    # Check ports
    check_ports()
    
    # Check environment variables
    print("\n🔧 Environment Variables:")
    env_vars = [
        "VITALS_WEBSOCKET_URI",
        "VITALS_API_KEY", 
        "VITALS_AUTH_TOKEN"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if 'KEY' in var or 'TOKEN' in var:
                masked = value[:4] + "*" * (len(value) - 8) + value[-4:] if len(value) > 8 else "*" * len(value)
                print(f"✅ {var}: {masked}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
    
    # Check services
    print("\n🏥 Service Health Checks:")
    
    services = [
        ("http://localhost:8002/inference/health", "Inference API", False),
        ("https://localhost:8001/health", "Live Stream Service (HTTPS)", False),
        ("http://localhost:8001/health", "Live Stream Service (HTTP)", False),
        ("https://localhost:8443", "HTTPS Server", False)
    ]
    
    results = []
    for url, name, verify_ssl in services:
        if url.startswith('https'):
            result = await check_service_async(url, name, use_ssl=not verify_ssl)
        else:
            result = check_service_sync(url, name, verify_ssl)
        results.append((name, result[0]))
    
    # Summary
    print("\n📊 Summary:")
    healthy_services = sum(1 for _, healthy in results if healthy)
    total_services = len(results)
    print(f"Services Running: {healthy_services}/{total_services}")
    
    if healthy_services == 0:
        print("\n💡 Troubleshooting Tips:")
        print("1. Start services with: python start_websocket_services.py")
        print("2. Check if SSL certificates exist: ls -la server.*")
        print("3. Generate SSL certificates if missing:")
        print("   openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt -days 365 -nodes")
        print("4. Check logs for detailed error messages")
        print("5. Ensure all required model files are present")
    
    elif healthy_services < total_services:
        print("\n⚠️  Some services are not running. Check the logs for details.")
    
    else:
        print("\n🎉 All services are healthy!")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Test Data Flow Between DV and HV Components
Tests that drowsiness_score and timestamp flow correctly from process_frame_inference() 
to synchronized_inference_engine and that HV data is properly synchronized.
"""

import time
import logging
import requests
import numpy as np
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dv_data_flow():
    """Test DV data flow through inference API."""
    print("🧪 Testing DV Data Flow")
    print("-" * 30)
    
    try:
        # Test inference API health
        response = requests.get("http://localhost:8002/inference/health", timeout=5)
        if response.status_code == 200:
            print("✅ Inference API is running")
        else:
            print("❌ Inference API not responding")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Inference API: {e}")
        return False
    
    try:
        # Test sync engine status in inference API
        response = requests.get("http://localhost:8002/inference/sync_engine_status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Inference API Sync Engine: {status['status']}")
            if status['status'] == 'active':
                print(f"   DV Buffer: {status['buffer_status']['dv_buffer_size']}")
                print(f"   HV Buffer: {status['buffer_status']['hv_buffer_size']}")
                print(f"   Total DV Received: {status['buffer_status']['stats']['total_dv_received']}")
                print(f"   Total HV Received: {status['buffer_status']['stats']['total_hv_received']}")
                print(f"   Total Synchronized: {status['buffer_status']['stats']['total_synchronized']}")
        else:
            print("⚠️  Inference API Sync Engine status unavailable")
    except Exception as e:
        print(f"⚠️  Cannot get Inference API sync engine status: {e}")
    
    return True

def test_hv_data_flow():
    """Test HV data flow through live stream service."""
    print("\n🧪 Testing HV Data Flow")
    print("-" * 30)
    
    try:
        # Test live stream service health
        response = requests.get("https://localhost:8001/health", timeout=5, verify=False)
        if response.status_code == 200:
            print("✅ Live Stream Service is running")
        else:
            print("❌ Live Stream Service not responding")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Live Stream Service: {e}")
        return False
    
    try:
        # Test sync engine status in live stream service
        response = requests.get("https://localhost:8001/sync_engine_status", timeout=5, verify=False)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Live Stream Service Sync Engine: {status['status']}")
            if status['status'] == 'active':
                print(f"   DV Buffer: {status['buffer_status']['dv_buffer_size']}")
                print(f"   HV Buffer: {status['buffer_status']['hv_buffer_size']}")
                print(f"   Total DV Received: {status['buffer_status']['stats']['total_dv_received']}")
                print(f"   Total HV Received: {status['buffer_status']['stats']['total_hv_received']}")
                print(f"   Total Synchronized: {status['buffer_status']['stats']['total_synchronized']}")
        else:
            print("⚠️  Live Stream Service Sync Engine status unavailable")
    except Exception as e:
        print(f"⚠️  Cannot get Live Stream Service sync engine status: {e}")
    
    return True

def test_hv_api_endpoint():
    """Test HV data API endpoint."""
    print("\n🧪 Testing HV Data API Endpoint")
    print("-" * 30)
    
    try:
        # Test sending HV data via API
        test_hv_data = {
            'hv_score': 0.75,
            'hr': 85,
            'spo2': 96,
            'hr_median': 85,
            'spo2_median': 96,
            'temperature': 28.5,
            'co2_level': 1200,
            'risk_level': 'moderate',
            'interpretation': 'Moderate health risk detected',
            'source': 'test_api'
        }
        
        payload = {
            "hv_data": test_hv_data,
            "timestamp": time.time(),
            "source": "test_data_flow"
        }
        
        response = requests.post(
            "https://localhost:8001/sync_engine/receive_hv_data", 
            json=payload, 
            timeout=5, 
            verify=False
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ HV data API endpoint working")
                print(f"   Timestamp: {result['timestamp']:.3f}")
                print(f"   Source: {result['source']}")
            else:
                print(f"⚠️  HV data API rejected data: {result.get('message')}")
        else:
            print(f"❌ HV data API endpoint failed: HTTP {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing HV data API: {e}")

def test_synchronization():
    """Test synchronization between DV and HV data."""
    print("\n🧪 Testing DV-HV Synchronization")
    print("-" * 30)
    
    # Get initial status
    try:
        response = requests.get("https://localhost:8001/sync_engine_status", timeout=5, verify=False)
        if response.status_code == 200:
            initial_status = response.json()
            initial_sync_count = initial_status['buffer_status']['stats']['total_synchronized']
            print(f"📊 Initial synchronized count: {initial_sync_count}")
        else:
            print("⚠️  Could not get initial sync status")
            return
    except Exception as e:
        print(f"❌ Error getting initial status: {e}")
        return
    
    # Send test HV data
    try:
        test_hv_data = {
            'hv_score': 0.65,
            'hr': 78,
            'spo2': 98,
            'hr_median': 78,
            'spo2_median': 98,
            'temperature': 27.8,
            'co2_level': 1100,
            'risk_level': 'low',
            'interpretation': 'Good health indicators',
            'source': 'sync_test'
        }
        
        current_time = time.time()
        payload = {
            "hv_data": test_hv_data,
            "timestamp": current_time,
            "source": "sync_test"
        }
        
        response = requests.post(
            "https://localhost:8001/sync_engine/receive_hv_data", 
            json=payload, 
            timeout=5, 
            verify=False
        )
        
        if response.status_code == 200:
            print("✅ Test HV data sent")
        else:
            print(f"❌ Failed to send test HV data: HTTP {response.status_code}")
            return
            
    except Exception as e:
        print(f"❌ Error sending test HV data: {e}")
        return
    
    # Wait a moment for processing
    time.sleep(3)
    
    # Check final status
    try:
        response = requests.get("https://localhost:8001/sync_engine_status", timeout=5, verify=False)
        if response.status_code == 200:
            final_status = response.json()
            final_sync_count = final_status['buffer_status']['stats']['total_synchronized']
            print(f"📊 Final synchronized count: {final_sync_count}")
            
            if final_sync_count > initial_sync_count:
                print("✅ Synchronization occurred!")
            else:
                print("⚠️  No new synchronization detected")
                print("   This could mean:")
                print("   - No DV data was available to synchronize with")
                print("   - Timestamps were too far apart")
                print("   - Data is waiting for timeout")
            
            # Show latest result if available
            if final_status.get('latest_result'):
                result = final_status['latest_result']
                print(f"📈 Latest Result:")
                print(f"   Health Score: {result['health_score']:.3f}")
                print(f"   DV Score: {result['dv_score']:.3f}")
                print(f"   HV Score: {result['hv_score']:.3f}")
                print(f"   Mode: {result['mode']}")
                
        else:
            print("⚠️  Could not get final sync status")
            
    except Exception as e:
        print(f"❌ Error getting final status: {e}")

def main():
    """Run all data flow tests."""
    print("🚗 Vigilant AI - Data Flow Test")
    print("=" * 50)
    print("Testing drowsiness_score and timestamp flow from process_frame_inference()")
    print("to synchronized_inference_engine and HV data synchronization.")
    print("=" * 50)
    
    # Test DV data flow
    dv_ok = test_dv_data_flow()
    
    # Test HV data flow
    hv_ok = test_hv_data_flow()
    
    # Test HV API endpoint
    test_hv_api_endpoint()
    
    # Test synchronization
    if dv_ok and hv_ok:
        test_synchronization()
    else:
        print("\n⚠️  Skipping synchronization test due to service issues")
    
    print("\n" + "=" * 50)
    print("🏁 Data Flow Test Complete")
    print("\n💡 Tips:")
    print("   - Make sure both websocket services are running")
    print("   - Check that vitals websocket processor is connected")
    print("   - Monitor logs for detailed synchronization information")
    print("   - Use /sync_engine_status endpoints to monitor real-time status")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test Imports Script
Tests if all required modules can be imported correctly.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test importing all required modules."""
    print("🧪 Testing Module Imports")
    print("=" * 40)
    
    # Add ai_module to path
    ai_module_path = os.path.join(os.path.dirname(__file__), 'ai_module')
    vision_model_path = os.path.join(ai_module_path, 'vision_model')
    
    if ai_module_path not in sys.path:
        sys.path.append(ai_module_path)
    if vision_model_path not in sys.path:
        sys.path.append(vision_model_path)
    
    # Test core Python modules
    core_modules = [
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('fastapi', 'FastAPI'),
        ('websockets', 'WebSockets'),
        ('aiohttp', 'aiohttp'),
        ('uvicorn', 'Uvicorn')
    ]
    
    print("📦 Testing Core Modules:")
    for module, name in core_modules:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name} - {e}")
    
    # Test AI modules
    print("\n🧠 Testing AI Modules:")
    ai_modules = [
        ('face_detection', 'Face Detection'),
        ('feature_extraction', 'Feature Extraction'),
        ('window_processing', 'Window Processing')
    ]
    
    for module, name in ai_modules:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name} - {e}")
    
    # Test inference API
    print("\n🔍 Testing Inference API:")
    try:
        from inference_api import process_frame_inference
        print("   ✅ Inference API imports")
    except ImportError as e:
        print(f"   ❌ Inference API - {e}")
    
    # Test synchronized inference engine
    print("\n⚙️ Testing Synchronized Inference Engine:")
    try:
        from inference.synchronized_inference_engine import SynchronizedInferenceEngine
        print("   ✅ Synchronized Inference Engine")
    except ImportError as e:
        print(f"   ❌ Synchronized Inference Engine - {e}")
    
    # Test vitals processing
    print("\n💓 Testing Vitals Processing:")
    try:
        from vitals_processing.vitals_processor import VitalsProcessor
        from vitals_processing.vitals_websocket_processor import VitalsWebSocketProcessor
        print("   ✅ Vitals Processing")
    except ImportError as e:
        print(f"   ❌ Vitals Processing - {e}")
    
    print("\n🎉 Import testing completed!")

if __name__ == "__main__":
    test_imports()

#!/usr/bin/env python3
"""
Quick Fix Script
Fixes common import and setup issues for the Vigilant AI system.
"""

import os
import sys
from pathlib import Path

def fix_import_paths():
    """Fix import path issues."""
    print("🔧 Fixing Import Paths")
    print("-" * 30)
    
    # Check if ai_module directory exists
    ai_module_path = Path(__file__).parent / "ai_module"
    if not ai_module_path.exists():
        print("❌ ai_module directory not found")
        return False
    
    # Check if vision_model directory exists
    vision_model_path = ai_module_path / "vision_model"
    if not vision_model_path.exists():
        print("❌ ai_module/vision_model directory not found")
        return False
    
    # Check if required files exist
    required_files = [
        "face_detection.py",
        "feature_extraction.py", 
        "window_processing.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not (vision_model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files found")
    return True

def test_imports():
    """Test if imports work after fix."""
    print("\n🧪 Testing Imports")
    print("-" * 30)
    
    try:
        # Add paths
        project_root = os.path.dirname(__file__)
        ai_module_path = os.path.join(project_root, 'ai_module')
        vision_model_path = os.path.join(ai_module_path, 'vision_model')
        
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        if ai_module_path not in sys.path:
            sys.path.append(ai_module_path)
        if vision_model_path not in sys.path:
            sys.path.append(vision_model_path)
        
        # Test AI module imports
        from face_detection import FaceLandmarkDetector
        print("✅ face_detection imported")
        
        from feature_extraction import DrowsinessFeatureExtractor
        print("✅ feature_extraction imported")
        
        from window_processing import SlidingWindowProcessor
        print("✅ window_processing imported")
        
        # Test inference API
        from inference_api import process_frame_inference
        print("✅ inference_api imported")
        
        # Test vitals processing
        from vitals_processing.vitals_processor import VitalsProcessor
        print("✅ vitals_processor imported")
        
        from vitals_processing.vitals_websocket_processor import VitalsWebSocketProcessor
        print("✅ vitals_websocket_processor imported")
        
        # Test synchronized inference engine
        from inference.synchronized_inference_engine import SynchronizedInferenceEngine
        print("✅ synchronized_inference_engine imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Main fix function."""
    print("🚗 Vigilant AI - Quick Fix")
    print("=" * 40)
    
    # Fix import paths
    if not fix_import_paths():
        print("\n❌ Cannot fix import paths - missing required files")
        return False
    
    # Test imports
    if not test_imports():
        print("\n❌ Imports still failing after fix")
        return False
    
    print("\n✅ All fixes applied successfully!")
    print("\n📋 Next steps:")
    print("   1. Run: python setup_and_run.py")
    print("   2. Or run: python start_integrated_system.py")
    
    return True

if __name__ == "__main__":
    main()

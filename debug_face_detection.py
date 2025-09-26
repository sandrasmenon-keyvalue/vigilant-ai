#!/usr/bin/env python3
"""
Debug script to test face detection step by step.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add ai-module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_module', 'vision_model'))

def test_basic_mediapipe():
    """Test basic MediaPipe face detection without our wrapper."""
    print("🧪 Testing basic MediaPipe face detection...")
    
    try:
        import mediapipe as mp
        
        # Initialize MediaPipe Face Landmarker
        mp_tasks = mp.tasks
        mp_vision = mp.tasks.vision
        
        # Download model if needed
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "face_landmarker.task"
        
        if not model_path.exists():
            print("📥 Downloading MediaPipe model...")
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, str(model_path))
            print(f"✅ Model downloaded to {model_path}")
        
        # Create detector
        base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )
        
        with mp_vision.FaceLandmarker.create_from_options(options) as detector:
            print("✅ MediaPipe detector created successfully")
            
            # Test with a sample image
            test_image_path = input("Enter path to test image: ").strip()
            
            if not test_image_path or not Path(test_image_path).exists():
                print("❌ Invalid image path")
                return False
            
            # Load and process image
            image = cv2.imread(test_image_path)
            if image is None:
                print("❌ Could not load image")
                return False
            
            print(f"📏 Image shape: {image.shape}")
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Detect
            result = detector.detect(mp_image)
            
            print(f"🔍 Detection result:")
            print(f"   Face landmarks: {len(result.face_landmarks) if result.face_landmarks else 0}")
            print(f"   Face blendshapes: {len(result.face_blendshapes) if result.face_blendshapes else 0}")
            
            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]
                print(f"   Total landmarks: {len(face_landmarks)}")
                print("✅ Face detection successful!")
                
                if result.face_blendshapes:
                    blendshapes = result.face_blendshapes[0]
                    print(f"   Total blendshapes: {len(blendshapes)}")
                    print("✅ Blendshape detection successful!")
                
                return True
            else:
                print("❌ No face detected")
                return False
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_our_wrapper():
    """Test our face detection wrapper."""
    print("\n🧪 Testing our face detection wrapper...")
    
    try:
        from face_landmarker_live import FaceLandmarkDetector
        
        # Initialize detector
        detector = FaceLandmarkDetector(
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            num_faces=1,
            output_blendshapes=True
        )
        
        print("✅ Our detector created successfully")
        
        # Test with a sample image
        test_image_path = input("Enter path to test image: ").strip()
        
        if not test_image_path or not Path(test_image_path).exists():
            print("❌ Invalid image path")
            return False
        
        # Load image
        image = cv2.imread(test_image_path)
        if image is None:
            print("❌ Could not load image")
            return False
        
        print(f"📏 Image shape: {image.shape}")
        
        # Detect landmarks
        landmarks = detector.detect_landmarks(image)
        
        if landmarks is None:
            print("❌ No landmarks returned")
            return False
        
        print(f"✅ Landmarks returned with keys: {list(landmarks.keys())}")
        
        if landmarks.get('face_detected', False):
            print("✅ Face detected successfully!")
            print(f"   📊 Total landmarks: {len(landmarks.get('all_landmarks', []))}")
            print(f"   🎭 Blendshapes available: {bool(landmarks.get('blendshapes'))}")
            if landmarks.get('blendshapes'):
                print(f"   🎭 Blendshapes count: {len(landmarks['blendshapes'])}")
            return True
        else:
            print("❌ Face not detected (face_detected=False)")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run debug tests."""
    print("🔧 FACE DETECTION DEBUG")
    print("=" * 30)
    
    # Test 1: Basic MediaPipe
    success1 = test_basic_mediapipe()
    
    # Test 2: Our wrapper
    success2 = test_our_wrapper()
    
    print("\n📊 RESULTS:")
    print(f"   Basic MediaPipe: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   Our Wrapper: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and not success2:
        print("\n🔍 DIAGNOSIS: MediaPipe works, but our wrapper has issues")
    elif not success1:
        print("\n🔍 DIAGNOSIS: MediaPipe itself is not working")
    elif success1 and success2:
        print("\n🔍 DIAGNOSIS: Everything is working correctly")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script to verify face detection is working properly.
"""

import sys
import os
import cv2
from pathlib import Path

# Add ai-module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_module', 'vision_model'))

from face_landmarker_live import FaceLandmarkDetector

def test_face_detection(image_path: str):
    """Test face detection on a single image."""
    print(f"🧪 Testing face detection on: {image_path}")
    
    # Initialize detector with lower confidence thresholds
    detector = FaceLandmarkDetector(
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        num_faces=1,
        output_blendshapes=True
    )
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return False
    
    print(f"📏 Image shape: {image.shape}")
    
    # Detect landmarks
    landmarks = detector.detect_landmarks(image)
    
    if landmarks is None:
        print("❌ No landmarks returned (landmarks is None)")
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

def main():
    """Test face detection on sample images."""
    print("🧪 FACE DETECTION TEST")
    print("=" * 30)
    
    # Get test image path
    test_image = input("Enter path to test image: ").strip()
    
    if not test_image:
        print("❌ No image path provided!")
        return
    
    if not Path(test_image).exists():
        print(f"❌ Image not found: {test_image}")
        return
    
    # Test face detection
    success = test_face_detection(test_image)
    
    if success:
        print("\n✅ Face detection is working correctly!")
    else:
        print("\n❌ Face detection failed!")
        print("Possible issues:")
        print("  1. Image quality too low")
        print("  2. Face not clearly visible")
        print("  3. Lighting conditions poor")
        print("  4. Face too small or too large")
        print("  5. MediaPipe model not properly loaded")

if __name__ == "__main__":
    main()

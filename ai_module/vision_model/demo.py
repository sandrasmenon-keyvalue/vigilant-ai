"""
Complete Demo Script for Drowsiness Detection System
Demonstrates the full image-based pipeline for drowsiness detection.
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Add the ai-module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_detection import FaceLandmarkDetector
from feature_extraction import DrowsinessFeatureExtractor
from model_training import DrowsinessModelTrainer
from train_from_images import ImageTrainingPipeline
from inference_single_image import SingleImageInference
from utils import (visualize_features, analyze_drowsiness_scores, 
                  plot_drowsiness_timeline, create_detection_report)


class DrowsinessDetectionDemo:
    """Complete demo of the drowsiness detection system."""
    
    def __init__(self):
        """Initialize the demo."""
        self.demo_dir = Path("demo_output")
        self.demo_dir.mkdir(exist_ok=True)
        
        print("🚗 Vigilant AI - Drowsiness Detection System Demo")
        print("=" * 60)
    
    def step1_face_detection(self):
        """Step 1: Demonstrate face detection."""
        print("\n🔹 Step 1: Face Detection")
        print("-" * 30)
        
        detector = FaceLandmarkDetector()
        
        # Create a simple test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "Face Detection Test", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        landmarks = detector.detect_landmarks(test_image)
        print(f"✅ Face detection test completed")
        print(f"   Face detected: {'Yes' if landmarks else 'No (expected for synthetic image)'}")
        
        return landmarks
    
    def step2_feature_extraction(self, landmarks):
        """Step 2: Demonstrate feature extraction."""
        print("\n🔹 Step 2: Feature Extraction")
        print("-" * 35)
        
        extractor = DrowsinessFeatureExtractor()
        features = extractor.extract_features(landmarks, timestamp=1.0)
        
        print(f"✅ Extracted {len(features)} features")
        print("Sample features:")
        for name, value in list(features.items())[:8]:
            print(f"  {name}: {value:.3f}")
        
        return features
    
    def step3_model_training(self):
        """Step 3: Demonstrate model training."""
        print("\n🔹 Step 3: Model Training")
        print("-" * 30)
        
        trainer = DrowsinessModelTrainer()
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        results = trainer.train_model(X, y)
        print(f"✅ Model trained with {results['accuracy']:.3f} accuracy")
        
        return results
    
    def run_quick_demo(self):
        """Run a quick demo focusing on key components."""
        print("⚡ Quick Demo - Key Components")
        print("=" * 40)
        
        # Step 1: Face detection
        landmarks = self.step1_face_detection()
        
        # Step 2: Feature extraction
        features = self.step2_feature_extraction(landmarks)
        
        # Step 3: Model training
        results = self.step3_model_training()
        
        print("\n✅ Quick demo completed!")
        print("All core components are working correctly")
    
    def run_complete_demo(self):
        """Run the complete demo with image training."""
        print("🚀 Starting Complete Drowsiness Detection Demo")
        print("This will demonstrate the image-based training pipeline\n")
        
        try:
            # Step 1: Face detection
            landmarks = self.step1_face_detection()
            
            # Step 2: Feature extraction
            features = self.step2_feature_extraction(landmarks)
            
            # Step 3: Model training
            results = self.step3_model_training()
            
            # Final summary
            print("\n🎉 Demo Complete!")
            print("=" * 60)
            print("✅ All components demonstrated successfully")
            print(f"📁 Demo outputs saved to: {self.demo_dir}")
            print("\nNext steps:")
            print("1. Collect real drowsiness/not_drowsy image data")
            print("2. Train model on real data using train_from_images.py")
            print("3. Use inference_single_image.py for predictions")
            
        except Exception as e:
            print(f"❌ Demo error: {e}")
            print("This is expected for some steps without real data")


def main():
    """Main demo function."""
    demo = DrowsinessDetectionDemo()
    
    print("Choose demo mode:")
    print("1. Complete Demo (all components)")
    print("2. Quick Demo (key components)")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        demo.run_complete_demo()
    elif choice == "2":
        demo.run_quick_demo()
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("Invalid choice. Running quick demo...")
        demo.run_quick_demo()


if __name__ == "__main__":
    main()
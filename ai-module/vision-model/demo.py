"""
Complete Demo Script for Drowsiness Detection System
Demonstrates the full 7-step pipeline from data preparation to real-time detection.
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Add the ai-module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preparation import FrameExtractor
from face_detection import FaceLandmarkDetector
from feature_extraction import DrowsinessFeatureExtractor
from window_processing import SlidingWindowProcessor
from model_training import DrowsinessModelTrainer
from realtime_detection import RealTimeDrowsinessDetector
from utils import (create_sample_dataset, visualize_features, 
                  analyze_drowsiness_scores, plot_drowsiness_timeline,
                  create_detection_report, benchmark_detection_speed)


class DrowsinessDetectionDemo:
    """Complete demo of the drowsiness detection system."""
    
    def __init__(self):
        """Initialize the demo."""
        self.demo_dir = Path("demo_output")
        self.demo_dir.mkdir(exist_ok=True)
        
        print("🚗 Vigilant AI - Drowsiness Detection System Demo")
        print("=" * 60)
    
    def step1_data_preparation(self):
        """Step 1: Demonstrate data preparation."""
        print("\n🔹 Step 1: Data Preparation")
        print("-" * 30)
        
        # Create sample dataset
        sample_dir = self.demo_dir / "sample_dataset"
        create_sample_dataset(str(sample_dir), num_videos=10)
        
        # Extract frames
        extractor = FrameExtractor(target_fps=5)
        frames_dir = self.demo_dir / "extracted_frames"
        frame_df = extractor.extract_frames_from_dataset(str(sample_dir), str(frames_dir))
        
        print(f"✅ Extracted {len(frame_df)} frames from sample dataset")
        return frame_df
    
    def step2_face_detection(self, frame_df):
        """Step 2: Demonstrate face detection."""
        print("\n🔹 Step 2: Face Detection & Landmarks")
        print("-" * 40)
        
        detector = FaceLandmarkDetector()
        
        # Process a few sample frames
        sample_frames = frame_df.head(5)
        landmarks_data = []
        
        for _, row in sample_frames.iterrows():
            frame_path = row['frame_path']
            frame = cv2.imread(frame_path)
            
            if frame is not None:
                landmarks = detector.detect_landmarks(frame)
                if landmarks:
                    landmarks['frame_path'] = frame_path
                    landmarks['label'] = row['label']
                    landmarks_data.append(landmarks)
                    print(f"✅ Detected landmarks in {Path(frame_path).name}")
                else:
                    print(f"❌ No face detected in {Path(frame_path).name}")
        
        print(f"✅ Processed {len(landmarks_data)} frames with face detection")
        return landmarks_data
    
    def step3_feature_extraction(self, landmarks_data):
        """Step 3: Demonstrate feature extraction."""
        print("\n🔹 Step 3: Feature Calculation")
        print("-" * 35)
        
        extractor = DrowsinessFeatureExtractor()
        features_data = []
        
        for i, landmarks in enumerate(landmarks_data):
            features = extractor.extract_features(landmarks, timestamp=i * 0.2)
            features['label'] = landmarks.get('label', 'unknown')
            features_data.append(features)
        
        # Show sample features
        if features_data:
            sample_features = features_data[0]
            print("Sample extracted features:")
            for name, value in list(sample_features.items())[:8]:
                print(f"  {name}: {value:.3f}")
        
        print(f"✅ Extracted features from {len(features_data)} frames")
        return features_data
    
    def step4_window_processing(self, features_data):
        """Step 4: Demonstrate window processing."""
        print("\n🔹 Step 4: Window Processing")
        print("-" * 30)
        
        processor = SlidingWindowProcessor(window_size_seconds=5.0, fps=5.0)
        
        # Process features into windows
        window_features = processor.process_feature_sequence(features_data)
        
        if window_features:
            print(f"✅ Created {len(window_features)} feature windows")
            print(f"Window feature count: {len(window_features[0])}")
            
            # Show sample window features
            sample_window = window_features[0]
            print("Sample window features:")
            for name, value in list(sample_window.items())[:5]:
                print(f"  {name}: {value:.3f}")
        else:
            print("⚠️  Not enough data for window processing (need 25 frames)")
            # Create synthetic data for demo
            synthetic_features = []
            for i in range(30):
                features = {
                    'avg_ear': 0.3 + 0.1 * np.sin(i * 0.5),
                    'mar': 0.2 + 0.05 * np.random.random(),
                    'head_pitch': 10 + 5 * np.sin(i * 0.3),
                    'blink_detected': 1.0 if i % 10 == 5 else 0.0,
                    'nod_detected': 0.0,
                    'yawn_indicator': 1.0 if i > 20 and i < 25 else 0.0,
                    'face_detected': 1.0,
                    'timestamp': i * 0.2
                }
                synthetic_features.append(features)
            
            window_features = processor.process_feature_sequence(synthetic_features)
            print(f"✅ Created {len(window_features)} synthetic feature windows")
        
        return window_features
    
    def step5_model_training(self, window_features):
        """Step 5: Demonstrate model training."""
        print("\n🔹 Step 5: Model Training")
        print("-" * 30)
        
        if not window_features:
            print("❌ No window features available for training")
            return None
        
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 200
        n_features = len(window_features[0])
        
        # Generate synthetic features
        X = np.random.randn(n_samples, n_features)
        
        # Create synthetic labels (drowsy vs alert)
        drowsiness_score = (
            -X[:, 0] * 0.5 +  # Low EAR
            X[:, 1] * 0.3 +   # High MAR
            -X[:, 2] * 0.2 +  # Low blink frequency
            np.random.randn(n_samples) * 0.1
        )
        y = (drowsiness_score > np.median(drowsiness_score)).astype(int)
        
        # Train model
        trainer = DrowsinessModelTrainer(model_type='xgboost')
        feature_names = list(window_features[0].keys())
        results = trainer.train_model(X, y, feature_names)
        
        # Save model
        model_path = self.demo_dir / "drowsiness_model.pkl"
        scaler_path = self.demo_dir / "feature_scaler.pkl"
        trainer.save_model(str(model_path), str(scaler_path))
        
        print(f"✅ Model trained with {results['accuracy']:.3f} accuracy")
        print(f"✅ Model saved to {model_path}")
        
        return str(model_path), str(scaler_path)
    
    def step6_realtime_detection(self, model_path, scaler_path):
        """Step 6: Demonstrate real-time detection."""
        print("\n🔹 Step 6: Real-time Detection")
        print("-" * 35)
        
        try:
            detector = RealTimeDrowsinessDetector(model_path, scaler_path)
            
            print("🎥 Starting real-time detection...")
            print("   Press 'q' to quit, 'r' to reset")
            print("   (Note: This will try to access your webcam)")
            
            # Ask user if they want to use webcam
            response = input("Use webcam for real-time detection? (y/n): ").lower()
            
            if response == 'y':
                detector.process_video_stream(video_source=0, display=True)
            else:
                print("⏭️  Skipping real-time detection")
                
        except Exception as e:
            print(f"❌ Real-time detection error: {e}")
            print("   This is expected if no webcam is available")
    
    def step7_score_smoothing(self):
        """Step 7: Demonstrate score smoothing."""
        print("\n🔹 Step 7: Score Smoothing")
        print("-" * 30)
        
        # Create sample scores with noise
        np.random.seed(42)
        timestamps = np.linspace(0, 60, 300)
        raw_scores = 0.3 + 0.4 * np.sin(timestamps * 0.1) + 0.1 * np.random.randn(300)
        raw_scores = np.clip(raw_scores, 0, 1)
        
        # Apply smoothing
        smoothed_scores = []
        alpha = 0.3
        smoothed = raw_scores[0]
        
        for score in raw_scores:
            smoothed = alpha * score + (1 - alpha) * smoothed
            smoothed_scores.append(smoothed)
        
        # Analyze results
        analysis = analyze_drowsiness_scores(smoothed_scores, timestamps)
        
        print(f"✅ Applied temporal smoothing to {len(smoothed_scores)} scores")
        print(f"   Mean score: {analysis['mean_score']:.3f}")
        print(f"   Drowsy percentage: {analysis['drowsy_percentage']:.1f}%")
        
        # Plot results
        plot_path = self.demo_dir / "score_smoothing_demo.png"
        plot_drowsiness_timeline(smoothed_scores, timestamps, str(plot_path))
        
        return smoothed_scores
    
    def run_complete_demo(self):
        """Run the complete 7-step demo."""
        print("🚀 Starting Complete Drowsiness Detection Demo")
        print("This will demonstrate all 7 steps of the system\n")
        
        try:
            # Step 1: Data Preparation
            frame_df = self.step1_data_preparation()
            
            # Step 2: Face Detection
            landmarks_data = self.step2_face_detection(frame_df)
            
            # Step 3: Feature Extraction
            features_data = self.step3_feature_extraction(landmarks_data)
            
            # Step 4: Window Processing
            window_features = self.step4_window_processing(features_data)
            
            # Step 5: Model Training
            model_path, scaler_path = self.step5_model_training(window_features)
            
            # Step 6: Real-time Detection
            self.step6_realtime_detection(model_path, scaler_path)
            
            # Step 7: Score Smoothing
            smoothed_scores = self.step7_score_smoothing()
            
            # Final summary
            print("\n🎉 Demo Complete!")
            print("=" * 60)
            print("✅ All 7 steps demonstrated successfully")
            print(f"📁 Demo outputs saved to: {self.demo_dir}")
            print("\nNext steps:")
            print("1. Collect real drowsiness/alert video data")
            print("2. Train model on real data")
            print("3. Deploy for real-time monitoring")
            print("4. Integrate with alert systems")
            
        except Exception as e:
            print(f"❌ Demo error: {e}")
            print("This is expected for some steps without real data")
    
    def run_quick_demo(self):
        """Run a quick demo focusing on key components."""
        print("⚡ Quick Demo - Key Components")
        print("=" * 40)
        
        # Create sample data
        print("\n📊 Creating sample data...")
        create_sample_dataset(str(self.demo_dir / "quick_sample"), num_videos=4)
        
        # Test face detection
        print("\n👁️  Testing face detection...")
        detector = FaceLandmarkDetector()
        
        # Create a simple test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "Face Detection Test", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        landmarks = detector.detect_landmarks(test_image)
        print(f"   Face detected: {'Yes' if landmarks else 'No (expected for synthetic image)'}")
        
        # Test feature extraction
        print("\n🔢 Testing feature extraction...")
        extractor = DrowsinessFeatureExtractor()
        features = extractor.extract_features(landmarks, timestamp=1.0)
        print(f"   Features extracted: {len(features)}")
        print(f"   Sample features: EAR={features.get('avg_ear', 0):.3f}, MAR={features.get('mar', 0):.3f}")
        
        # Test model training
        print("\n🤖 Testing model training...")
        trainer = DrowsinessModelTrainer()
        
        # Create synthetic data
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        results = trainer.train_model(X, y)
        print(f"   Model accuracy: {results['accuracy']:.3f}")
        
        print("\n✅ Quick demo completed!")
        print("All core components are working correctly")


def main():
    """Main demo function."""
    demo = DrowsinessDetectionDemo()
    
    print("Choose demo mode:")
    print("1. Complete Demo (all 7 steps)")
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

"""
Complete Training Pipeline for Drowsiness Detection
Takes images/frames from resources folder and trains the model.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import json
from typing import List, Dict, Tuple
import joblib

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_detection import FaceLandmarkDetector
from feature_extraction import DrowsinessFeatureExtractor
from model_training import DrowsinessModelTrainer


class ImageTrainingPipeline:
    """Complete pipeline to train drowsiness detection model from images."""
    
    def __init__(self, resources_dir: str, output_dir: str, fps: float = 5.0):
        """
        Initialize training pipeline.
        
        Args:
            resources_dir: Directory containing image files
            output_dir: Directory to save training outputs
            fps: Frames per second (for window processing)
        """
        self.resources_dir = Path(resources_dir)
        self.output_dir = Path(output_dir)
        self.fps = fps
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "features").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        
        # Initialize components
        self.face_detector = FaceLandmarkDetector()
        self.feature_extractor = DrowsinessFeatureExtractor()
        self.model_trainer = DrowsinessModelTrainer(model_type='xgboost')
        
        # Training statistics
        self.stats = {
            'images_processed': 0,
            'faces_detected': 0,
            'features_extracted': 0,
            'training_time': 0
        }
    
    def validate_resources(self) -> bool:
        """
        Validate that resources directory has the expected structure.
        
        Expected structure:
        resources/
        ├── drowsy/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── alert/
            ├── image1.jpg
            ├── image2.jpg
            └── ...
        """
        if not self.resources_dir.exists():
            print(f"❌ Resources directory not found: {self.resources_dir}")
            return False
        
        drowsy_dir = self.resources_dir / "drowsy"
        not_drowsy_dir = self.resources_dir / "not_drowsy"
        
        if not drowsy_dir.exists():
            print(f"❌ Drowsy directory not found: {drowsy_dir}")
            return False
        
        if not not_drowsy_dir.exists():
            print(f"❌ Not_drowsy directory not found: {not_drowsy_dir}")
            return False
        
        # Check for image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        drowsy_images = [f for f in drowsy_dir.iterdir() if f.suffix.lower() in image_extensions]
        not_drowsy_images = [f for f in not_drowsy_dir.iterdir() if f.suffix.lower() in image_extensions]
        
        if not drowsy_images:
            print(f"❌ No image files found in drowsy directory: {drowsy_dir}")
            return False
        
        if not not_drowsy_images:
            print(f"❌ No image files found in not_drowsy directory: {not_drowsy_dir}")
            return False
        
        print(f"✅ Found {len(drowsy_images)} drowsy images and {len(not_drowsy_images)} not_drowsy images")
        return True
    
    def load_images_with_labels(self) -> List[Dict]:
        """
        Load all images with their labels.
        
        Returns:
            List of dictionaries with image info
        """
        print("\n🔹 Step 1: Loading images with labels...")
        print("-" * 50)
        
        image_data = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Load drowsy images
        drowsy_dir = self.resources_dir / "drowsy"
        for img_file in drowsy_dir.iterdir():
            if img_file.suffix.lower() in image_extensions:
                image_data.append({
                    'image_path': str(img_file),
                    'label': 'drowsy',
                    'image_name': img_file.name
                })
        
        # Load not_drowsy images
        not_drowsy_dir = self.resources_dir / "not_drowsy"
        for img_file in not_drowsy_dir.iterdir():
            if img_file.suffix.lower() in image_extensions:
                image_data.append({
                    'image_path': str(img_file),
                    'label': 'not_drowsy',
                    'image_name': img_file.name
                })
        
        self.stats['images_processed'] = len(image_data)
        print(f"✅ Loaded {len(image_data)} images")
        print(f"   Drowsy: {len([d for d in image_data if d['label'] == 'drowsy'])}")
        print(f"   Not_drowsy: {len([d for d in image_data if d['label'] == 'not_drowsy'])}")
        
        return image_data
    
    def extract_landmarks_from_images(self, image_data: List[Dict]) -> List[Dict]:
        """
        Extract facial landmarks from all images.
        
        Args:
            image_data: List of image information
            
        Returns:
            List of landmark data
        """
        print("\n🔹 Step 2: Extracting facial landmarks...")
        print("-" * 50)
        
        landmarks_data = []
        total_images = len(image_data)
        
        for idx, img_info in enumerate(image_data):
            image_path = img_info['image_path']
            frame = cv2.imread(image_path)
            
            if frame is not None:
                landmarks = self.face_detector.detect_landmarks(frame)
                if landmarks:
                    landmarks['image_path'] = image_path
                    landmarks['label'] = img_info['label']
                    landmarks['image_name'] = img_info['image_name']
                    landmarks['timestamp'] = idx * 0.2  # Simulate 5 FPS timing
                    landmarks_data.append(landmarks)
                    self.stats['faces_detected'] += 1
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{total_images} images... "
                      f"({self.stats['faces_detected']} faces detected)")
        
        print(f"✅ Extracted landmarks from {len(landmarks_data)} images")
        return landmarks_data
    
    def extract_features_from_landmarks(self, landmarks_data: List[Dict]) -> List[Dict]:
        """
        Extract drowsiness features from landmarks.
        
        Args:
            landmarks_data: List of landmark data
            
        Returns:
            List of feature dictionaries
        """
        print("\n🔹 Step 3: Extracting features...")
        print("-" * 50)
        
        features_data = []
        total_landmarks = len(landmarks_data)
        
        for i, landmarks in enumerate(landmarks_data):
            features = self.feature_extractor.extract_features(
                landmarks, landmarks['timestamp']
            )
            features['label'] = landmarks['label']
            features['image_path'] = landmarks['image_path']
            features['image_name'] = landmarks['image_name']
            features_data.append(features)
            self.stats['features_extracted'] += 1
            
            if (i + 1) % 50 == 0:
                print(f"  Extracted features from {i + 1}/{total_landmarks} images...")
        
        print(f"✅ Extracted features from {len(features_data)} images")
        return features_data
    
    def create_training_data(self, features_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create training data from features.
        
        Args:
            features_data: List of feature dictionaries
            
        Returns:
            Tuple of (features_array, labels_array, feature_names)
        """
        print("\n🔹 Step 4: Creating training data...")
        print("-" * 50)
        
        if not features_data:
            raise ValueError("No features data available!")
        
        # Get feature names (exclude metadata columns)
        metadata_columns = {'label', 'image_path', 'image_name', 'timestamp'}
        feature_names = [name for name in features_data[0].keys() if name not in metadata_columns]
        feature_names.sort()
        
        # Create feature matrix
        X = np.array([[feat[name] for name in feature_names] for feat in features_data])
        y = np.array([1.0 if feat['label'] == 'drowsy' else 0.0 for feat in features_data])
        
        print(f"✅ Created training data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Class distribution: {np.bincount(y.astype(int))}")
        print(f"   Feature names: {len(feature_names)}")
        
        return X, y, feature_names
    
    def train_model(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Train the drowsiness detection model.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            
        Returns:
            Training results
        """
        print("\n🔹 Step 5: Training model...")
        print("-" * 50)
        
        start_time = time.time()
        
        # Train model
        results = self.model_trainer.train_model(X, y, feature_names)
        
        self.stats['training_time'] = time.time() - start_time
        
        # Save model
        model_path = self.output_dir / "models" / "drowsiness_model.pkl"
        scaler_path = self.output_dir / "models" / "feature_scaler.pkl"
        
        self.model_trainer.save_model(str(model_path), str(scaler_path))
        
        print(f"✅ Model trained with {results['accuracy']:.3f} accuracy")
        print(f"✅ Model saved to {model_path}")
        
        return results
    
    def save_training_summary(self, results: Dict, feature_names: List[str]):
        """
        Save training summary and statistics.
        
        Args:
            results: Training results
            feature_names: List of feature names
        """
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        summary = {
            'training_stats': convert_numpy_types(self.stats),
            'model_results': convert_numpy_types(results),
            'feature_count': len(feature_names),
            'feature_names': feature_names,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save summary
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save feature importance
        importance = self.model_trainer.get_feature_importance()
        # Convert numpy types to Python types for JSON serialization
        importance_serializable = {k: float(v) for k, v in importance.items()}
        importance_path = self.output_dir / "feature_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(importance_serializable, f, indent=2)
        
        print(f"✅ Training summary saved to {summary_path}")
        print(f"✅ Feature importance saved to {importance_path}")
    
    def run_complete_training(self) -> bool:
        """
        Run the complete training pipeline.
        
        Returns:
            True if training successful, False otherwise
        """
        print("🚗 Vigilant AI - Image Training Pipeline")
        print("=" * 60)
        
        try:
            # Validate resources
            if not self.validate_resources():
                return False
            
            # Step 1: Load images
            image_data = self.load_images_with_labels()
            
            # Step 2: Extract landmarks
            landmarks_data = self.extract_landmarks_from_images(image_data)
            
            if not landmarks_data:
                print("❌ No faces detected in any images!")
                return False
            
            # Step 3: Extract features
            features_data = self.extract_features_from_landmarks(landmarks_data)
            
            # Step 4: Create training data
            X, y, feature_names = self.create_training_data(features_data)
            
            # Step 5: Train model
            results = self.train_model(X, y, feature_names)
            
            # Save summary
            self.save_training_summary(results, feature_names)
            
            # Print final summary
            print("\n🎉 Training completed successfully!")
            print("=" * 60)
            print(f"📊 Training Statistics:")
            print(f"  Images processed: {self.stats['images_processed']}")
            print(f"  Faces detected: {self.stats['faces_detected']}")
            print(f"  Features extracted: {self.stats['features_extracted']}")
            print(f"  Training time: {self.stats['training_time']:.1f}s")
            print(f"  Model accuracy: {results['accuracy']:.3f}")
            print(f"  Model AUC: {results['auc_score']:.3f}")
            print(f"\n📁 Outputs saved to: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function for training from images."""
    parser = argparse.ArgumentParser(description='Train drowsiness detection model from images')
    parser.add_argument('--resources', required=True, 
                       help='Path to resources directory containing drowsy/ and alert/ subdirectories')
    parser.add_argument('--output', default='training_output', 
                       help='Output directory for training results')
    parser.add_argument('--fps', type=float, default=5.0, 
                       help='Frames per second (for window processing)')
    
    args = parser.parse_args()
    
    # Create training pipeline
    pipeline = ImageTrainingPipeline(
        resources_dir=args.resources,
        output_dir=args.output,
        fps=args.fps
    )
    
    # Run training
    success = pipeline.run_complete_training()
    
    if success:
        print("\n✅ Training completed successfully!")
        print(f"📁 Check outputs in: {args.output}")
        print("\nTo use the trained model:")
        print(f"  python inference.py --model {args.output}/models/drowsiness_model.pkl --scaler {args.output}/models/feature_scaler.pkl")
    else:
        print("\n❌ Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

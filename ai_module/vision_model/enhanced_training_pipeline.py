"""
Enhanced Training Pipeline for Drowsiness Detection
Uses MediaPipe Face Landmarker with optimized drowsiness features and blendshapes.
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
from typing import List, Dict, Tuple, Optional
import joblib
import logging
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_landmarker_live import FaceLandmarkDetector
from drowsiness_optimized_extraction import DrowsinessOptimizedExtractor
from model_training import DrowsinessModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTrainingPipeline:
    """
    Enhanced training pipeline using MediaPipe Face Landmarker and optimized drowsiness features.
    """
    
    def __init__(self, resources_dir: str, output_dir: str, model_type: str = 'xgboost'):
        """
        Initialize enhanced training pipeline.
        
        Args:
            resources_dir: Directory containing image files or video data
            output_dir: Directory to save training outputs
            model_type: Type of model to train ('xgboost', 'logistic', 'random_forest')
        """
        self.resources_dir = Path(resources_dir)
        self.output_dir = Path(output_dir)
        self.model_type = model_type
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "features").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        
        # Initialize enhanced components
        logger.info("🚀 Initializing MediaPipe Face Landmarker...")
        self.face_detector = FaceLandmarkDetector(
            min_face_detection_confidence=0.3,  # Lower threshold for better detection
            min_face_presence_confidence=0.3,   # Lower threshold for better detection
            min_tracking_confidence=0.3,        # Lower threshold for better detection
            num_faces=1,
            output_blendshapes=True  # Enable blendshapes for enhanced features
        )
        
        logger.info("🔧 Initializing enhanced drowsiness extractor...")
        self.feature_extractor = DrowsinessOptimizedExtractor(
            ear_threshold=0.25,
            mar_threshold=0.5
        )
        
        logger.info("🤖 Initializing model trainer...")
        self.model_trainer = DrowsinessModelTrainer(model_type=model_type)
        
        # Training statistics
        self.stats = {
            'images_processed': 0,
            'faces_detected': 0,
            'features_extracted': 0,
            'blendshapes_detected': 0,
            'training_time': 0,
            'model_type': model_type
        }
        
        logger.info("✅ Enhanced training pipeline initialized successfully!")
    
    def validate_resources(self) -> bool:
        """
        Validate that resources directory has the expected structure.
        
        Expected structure:
        resources/
        ├── drowsy/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── alert/  (or not_drowsy)
            ├── image1.jpg
            ├── image2.jpg
            └── ...
        """
        logger.info("🔍 Validating resources directory structure...")
        
        if not self.resources_dir.exists():
            logger.error(f"❌ Resources directory not found: {self.resources_dir}")
            return False
        
        drowsy_dir = self.resources_dir / "drowsy"
        
        if not drowsy_dir.exists():
            logger.error(f"❌ Drowsy directory not found: {drowsy_dir}")
            return False
        
        # Find non-drowsy directory (try multiple variations)
        non_drowsy_dir = None
        label_name = None
        
        possible_dirs = ["non_drowsy", "not_drowsy", "notdrowsy", "alert"]
        for dir_name in possible_dirs:
            test_dir = self.resources_dir / dir_name
            if test_dir.exists():
                non_drowsy_dir = test_dir
                label_name = dir_name
                break
        
        if non_drowsy_dir is None:
            logger.error(f"❌ Non-drowsy directory not found in {self.resources_dir}")
            logger.error(f"   Tried: {possible_dirs}")
            return False
        
        # Check for image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        drowsy_images = [f for f in drowsy_dir.iterdir() if f.suffix.lower() in image_extensions]
        non_drowsy_images = [f for f in non_drowsy_dir.iterdir() if f.suffix.lower() in image_extensions]
        
        if not drowsy_images:
            logger.error(f"❌ No image files found in drowsy directory: {drowsy_dir}")
            return False
        
        if not non_drowsy_images:
            logger.error(f"❌ No image files found in {label_name} directory: {non_drowsy_dir}")
            return False
        
        logger.info(f"✅ Found {len(drowsy_images)} drowsy images and {len(non_drowsy_images)} {label_name} images")
        return True
    
    def load_images_with_labels(self) -> List[Dict]:
        """
        Load all images with their labels using enhanced directory detection.
        
        Returns:
            List of dictionaries with image info
        """
        logger.info("📂 Loading images with labels...")
        
        image_data = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Load drowsy images
        drowsy_dir = self.resources_dir / "drowsy"
        for img_file in drowsy_dir.iterdir():
            if img_file.suffix.lower() in image_extensions:
                image_data.append({
                    'image_path': str(img_file),
                    'label': 'drowsy',
                    'image_name': img_file.name,
                    'binary_label': 1  # 1 for drowsy
                })
        
        # Load non-drowsy images
        not_drowsy_dir = self.resources_dir / "non_drowsy"  # Fixed directory name
        non_drowsy_label = "not_drowsy"
        
        if not not_drowsy_dir.exists():
            # Also try other variations
            alt_dirs = ["not_drowsy", "notdrowsy", "alert"]
            for alt_name in alt_dirs:
                alt_dir = self.resources_dir / alt_name
                if alt_dir.exists():
                    not_drowsy_dir = alt_dir
                    non_drowsy_label = alt_name
                    break
            else:
                raise ValueError(f"Non-drowsy directory not found! Tried: {alt_dirs}")
        
        for img_file in not_drowsy_dir.iterdir():
            if img_file.suffix.lower() in image_extensions:
                image_data.append({
                    'image_path': str(img_file),
                    'label': non_drowsy_label,
                    'image_name': img_file.name,
                    'binary_label': 0  # 0 for not drowsy
                })
        
        # Shuffle for better training
        np.random.shuffle(image_data)
        
        self.stats['images_processed'] = len(image_data)
        drowsy_count = len([d for d in image_data if d['binary_label'] == 1])
        alert_count = len([d for d in image_data if d['binary_label'] == 0])
        
        logger.info(f"✅ Loaded {len(image_data)} images")
        logger.info(f"   💤 Drowsy: {drowsy_count}")
        logger.info(f"   😊 Alert: {alert_count}")
        logger.info(f"   📊 Class balance: {drowsy_count/(drowsy_count+alert_count):.2%} drowsy")
        
        return image_data
    
    def extract_enhanced_features_from_images(self, image_data: List[Dict]) -> List[Dict]:
        """
        Extract enhanced features using MediaPipe Face Landmarker and blendshapes.
        
        Args:
            image_data: List of image information
            
        Returns:
            List of enhanced feature data
        """
        logger.info("🎯 Extracting enhanced features from images...")
        
        features_data = []
        total_images = len(image_data)
        failed_detections = 0
        
        for idx, img_info in enumerate(image_data):
            try:
                # Load image
                image_path = img_info['image_path']
                frame = cv2.imread(image_path)
                
                if frame is None:
                    logger.warning(f"⚠️ Could not load image: {image_path}")
                    continue
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect landmarks using MediaPipe Face Landmarker
                landmarks = self.face_detector.detect_landmarks(rgb_frame)
                
                # Debug: Print detection results for first few images
                if idx < 5:
                    if landmarks is None:
                        logger.warning(f"🔍 DEBUG: No landmarks returned for image {idx+1}: {img_info['image_name']}")
                    elif not landmarks.get('face_detected', False):
                        logger.warning(f"🔍 DEBUG: Face not detected for image {idx+1}: {img_info['image_name']}")
                        logger.warning(f"   Landmarks keys: {list(landmarks.keys()) if landmarks else 'None'}")
                    else:
                        logger.info(f"✅ DEBUG: Face detected for image {idx+1}: {img_info['image_name']}")
                
                if landmarks and landmarks.get('face_detected', False):
                    # Extract enhanced features
                    timestamp = idx * 0.2  # Simulate timing
                    
                    # Get blendshapes if available
                    blendshapes = landmarks.get('blendshapes', {})
                    if blendshapes:
                        self.stats['blendshapes_detected'] += 1
                    
                    # Extract features using optimized extractor
                    features = self.feature_extractor.extract_features(
                        landmarks, 
                        timestamp,
                        blendshapes=blendshapes
                    )
                    
                    # Add metadata
                    features.update({
                        'label': img_info['label'],
                        'binary_label': img_info['binary_label'],
                        'image_path': image_path,
                        'image_name': img_info['image_name'],
                        'timestamp': timestamp,
                        'has_blendshapes': bool(blendshapes)
                    })
                    
                    features_data.append(features)
                    self.stats['faces_detected'] += 1
                    self.stats['features_extracted'] += 1
                    
                else:
                    failed_detections += 1
                    logger.debug(f"❌ No face detected in: {img_info['image_name']}")
                
                # Progress update
                if (idx + 1) % 50 == 0 or (idx + 1) == total_images:
                    progress = (idx + 1) / total_images * 100
                    logger.info(f"  📊 Progress: {idx + 1}/{total_images} ({progress:.1f}%) - "
                              f"Faces: {self.stats['faces_detected']}, "
                              f"Blendshapes: {self.stats['blendshapes_detected']}")
            
            except Exception as e:
                logger.error(f"❌ Error processing {img_info.get('image_name', 'unknown')}: {e}")
                failed_detections += 1
                continue
        
        success_rate = len(features_data) / total_images * 100
        blendshape_rate = self.stats['blendshapes_detected'] / max(len(features_data), 1) * 100
        
        logger.info(f"✅ Enhanced feature extraction completed!")
        logger.info(f"   📈 Success rate: {success_rate:.1f}% ({len(features_data)}/{total_images})")
        logger.info(f"   🎭 Blendshapes detected: {blendshape_rate:.1f}% ({self.stats['blendshapes_detected']}/{len(features_data)})")
        logger.info(f"   ❌ Failed detections: {failed_detections}")
        
        return features_data
    
    def create_enhanced_training_data(self, features_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create training data from enhanced features.
        
        Args:
            features_data: List of enhanced feature dictionaries
            
        Returns:
            Tuple of (features_array, labels_array, feature_names)
        """
        logger.info("🔢 Creating enhanced training data...")
        
        if not features_data:
            raise ValueError("No features data available!")
        
        # Define metadata columns to exclude
        metadata_columns = {
            'label', 'binary_label', 'image_path', 'image_name', 
            'timestamp', 'has_blendshapes'
        }
        
        # Get feature names (exclude metadata)
        all_keys = set()
        for feat in features_data:
            all_keys.update(feat.keys())
        
        feature_names = sorted([name for name in all_keys if name not in metadata_columns])
        
        logger.info(f"📊 Enhanced features detected: {len(feature_names)}")
        logger.info(f"   📋 Feature list: {feature_names}")
        
        # Create feature matrix
        X = []
        y = []
        
        for feat in features_data:
            # Extract feature vector
            feature_vector = []
            for name in feature_names:
                value = feat.get(name, 0.0)  # Default to 0.0 if missing
                # Ensure numeric
                if isinstance(value, (int, float, np.number)):
                    feature_vector.append(float(value))
                else:
                    feature_vector.append(0.0)
            
            X.append(feature_vector)
            y.append(feat['binary_label'])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        # Validate data
        if np.any(np.isnan(X)):
            logger.warning("⚠️ NaN values detected in features, replacing with 0.0")
            X = np.nan_to_num(X, nan=0.0)
        
        if np.any(np.isinf(X)):
            logger.warning("⚠️ Infinite values detected in features, clipping")
            X = np.clip(X, -1e6, 1e6)
        
        # Class distribution
        class_counts = np.bincount(y)
        drowsy_count = class_counts[1] if len(class_counts) > 1 else 0
        alert_count = class_counts[0]
        
        logger.info(f"✅ Enhanced training data created!")
        logger.info(f"   📏 Shape: {X.shape[0]} samples × {X.shape[1]} features")
        logger.info(f"   🏷️  Labels: {alert_count} alert, {drowsy_count} drowsy")
        logger.info(f"   ⚖️  Balance: {drowsy_count/(drowsy_count+alert_count):.2%} drowsy")
        
        # Save feature statistics
        feature_stats = {
            'feature_names': feature_names,
            'feature_count': len(feature_names),
            'sample_count': X.shape[0],
            'class_distribution': {'alert': int(alert_count), 'drowsy': int(drowsy_count)},
            'feature_statistics': {}
        }
        
        for i, name in enumerate(feature_names):
            feature_stats['feature_statistics'][name] = {
                'mean': float(np.mean(X[:, i])),
                'std': float(np.std(X[:, i])),
                'min': float(np.min(X[:, i])),
                'max': float(np.max(X[:, i]))
            }
        
        # Save feature statistics
        stats_path = self.output_dir / "feature_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(feature_stats, f, indent=2)
        
        logger.info(f"📊 Feature statistics saved to {stats_path}")
        
        return X, y, feature_names
    
    def train_enhanced_model(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Train the enhanced drowsiness detection model.
        
        Args:
            X: Enhanced feature matrix
            y: Target labels
            feature_names: List of feature names
            
        Returns:
            Training results
        """
        logger.info(f"🚀 Training enhanced {self.model_type} model...")
        
        start_time = time.time()
        
        # Train model with enhanced features
        results = self.model_trainer.train_model(X, y, feature_names)
        
        training_time = time.time() - start_time
        self.stats['training_time'] = training_time
        
        # Save enhanced model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.output_dir / "models" / f"enhanced_drowsiness_model_{timestamp}.pkl"
        scaler_path = self.output_dir / "models" / f"enhanced_feature_scaler_{timestamp}.pkl"
        
        # Also save without timestamp for easy loading
        model_path_current = self.output_dir / "models" / "drowsiness_model.pkl"
        scaler_path_current = self.output_dir / "models" / "feature_scaler.pkl"
        
        self.model_trainer.save_model(str(model_path), str(scaler_path))
        self.model_trainer.save_model(str(model_path_current), str(scaler_path_current))
        
        logger.info(f"🎯 Enhanced model trained!")
        logger.info(f"   📊 Accuracy: {results['accuracy']:.3f}")
        logger.info(f"   📈 AUC Score: {results['auc_score']:.3f}")
        logger.info(f"   ⏱️  Training Time: {training_time:.1f}s")
        logger.info(f"   💾 Model saved to: {model_path_current}")
        
        return results
    
    def generate_training_report(self, results: Dict, feature_names: List[str]):
        """
        Generate comprehensive training report.
        """
        logger.info("📝 Generating training report...")
        
        # Get feature importance
        importance = self.model_trainer.get_feature_importance()
        
        # Create comprehensive report
        report = {
            'training_metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_type': self.model_type,
                'feature_extractor': 'DrowsinessOptimizedExtractor',
                'face_detector': 'MediaPipe Face Landmarker (478 landmarks + blendshapes)'
            },
            'training_statistics': self.stats,
            'model_performance': {
                'accuracy': results['accuracy'],
                'auc_score': results['auc_score'],
                'classification_report': results['classification_report']
            },
            'feature_analysis': {
                'feature_count': len(feature_names),
                'feature_names': feature_names,
                'feature_importance': importance
            },
            'data_quality': {
                'success_rate': self.stats['faces_detected'] / self.stats['images_processed'] * 100,
                'blendshape_coverage': self.stats['blendshapes_detected'] / max(self.stats['faces_detected'], 1) * 100
            }
        }
        
        # Save comprehensive report
        report_path = self.output_dir / "enhanced_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create human-readable summary
        summary_path = self.output_dir / "training_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("🚗 VIGILANT AI - Enhanced Training Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"📅 Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"🤖 Model Type: {self.model_type.upper()}\n")
            f.write(f"🎯 Feature Extractor: Enhanced with MediaPipe Face Landmarker\n\n")
            
            f.write("📊 PERFORMANCE METRICS:\n")
            f.write(f"   Accuracy: {results['accuracy']:.1%}\n")
            f.write(f"   AUC Score: {results['auc_score']:.3f}\n\n")
            
            f.write("📈 TRAINING STATISTICS:\n")
            f.write(f"   Images Processed: {self.stats['images_processed']}\n")
            f.write(f"   Faces Detected: {self.stats['faces_detected']}\n")
            f.write(f"   Success Rate: {self.stats['faces_detected']/self.stats['images_processed']:.1%}\n")
            f.write(f"   Blendshapes Detected: {self.stats['blendshapes_detected']}\n")
            f.write(f"   Training Time: {self.stats['training_time']:.1f}s\n\n")
            
            f.write("🎯 TOP 10 MOST IMPORTANT FEATURES:\n")
            for i, (feature, imp) in enumerate(list(importance.items())[:10]):
                f.write(f"   {i+1:2d}. {feature:<20}: {imp:.4f}\n")
            
            f.write("\n✅ Enhanced model ready for deployment!\n")
        
        logger.info(f"📋 Training report saved to: {report_path}")
        logger.info(f"📄 Summary saved to: {summary_path}")
    
    def run_enhanced_training(self) -> bool:
        """
        Run the complete enhanced training pipeline.
        
        Returns:
            True if training successful, False otherwise
        """
        logger.info("🚀 ENHANCED DROWSINESS DETECTION TRAINING")
        logger.info("=" * 60)
        
        try:
            # Step 1: Validate resources
            if not self.validate_resources():
                return False
            
            # Step 2: Load images with labels
            image_data = self.load_images_with_labels()
            if not image_data:
                logger.error("❌ No image data available!")
                return False
            
            # Step 3: Extract enhanced features
            features_data = self.extract_enhanced_features_from_images(image_data)
            if not features_data:
                logger.error("❌ No features extracted from images!")
                return False
            
            # Step 4: Create enhanced training data
            X, y, feature_names = self.create_enhanced_training_data(features_data)
            
            # Step 5: Train enhanced model
            results = self.train_enhanced_model(X, y, feature_names)
            
            # Step 6: Generate comprehensive report
            self.generate_training_report(results, feature_names)
            
            # Final success summary
            logger.info("\n🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info("📊 FINAL RESULTS:")
            logger.info(f"   🎯 Model Accuracy: {results['accuracy']:.1%}")
            logger.info(f"   📈 AUC Score: {results['auc_score']:.3f}")
            logger.info(f"   🔧 Features Used: {len(feature_names)} enhanced features")
            logger.info(f"   💾 Model Path: {self.output_dir}/models/drowsiness_model.pkl")
            logger.info(f"   📋 Report Path: {self.output_dir}/enhanced_training_report.json")
            
            logger.info("\n🚀 Your enhanced model is ready! Key improvements:")
            logger.info("   ✅ MediaPipe Face Landmarker (478 3D landmarks)")
            logger.info("   ✅ Blendshape facial expressions")
            logger.info("   ✅ Optimized drowsiness feature extraction")
            logger.info("   ✅ Better response to fatigue signs")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Enhanced training failed: {e}")
            import traceback
            logger.error("📋 Traceback:")
            logger.error(traceback.format_exc())
            return False


def main():
    """Main function for enhanced training."""
    parser = argparse.ArgumentParser(description='Train enhanced drowsiness detection model')
    parser.add_argument('--resources', required=True, 
                       help='Path to resources directory containing drowsy/ and alert/ subdirectories')
    parser.add_argument('--output', default='enhanced_training_output', 
                       help='Output directory for training results')
    parser.add_argument('--model', choices=['xgboost', 'logistic', 'random_forest'], 
                       default='xgboost', help='Model type to train')
    
    args = parser.parse_args()
    
    # Create enhanced training pipeline
    pipeline = EnhancedTrainingPipeline(
        resources_dir=args.resources,
        output_dir=args.output,
        model_type=args.model
    )
    
    # Run enhanced training
    success = pipeline.run_enhanced_training()
    
    if success:
        print("\n🎉 ENHANCED TRAINING SUCCESS!")
        print("=" * 50)
        print("Your new model features:")
        print("  🎯 MediaPipe Face Landmarker (478 landmarks)")
        print("  🎭 Facial expression blendshapes")
        print("  🔧 Optimized drowsiness detection")
        print("  📈 Better response to clear fatigue signs")
        print(f"\n📁 All outputs saved to: {args.output}")
        print(f"💾 Use this model: {args.output}/models/drowsiness_model.pkl")
        print("\nTo deploy the new model:")
        print("  1. Your inference_api.py will automatically use the new model")
        print("  2. Restart your services to load the enhanced model")
        print("  3. Test with live camera - should be much more responsive!")
    else:
        print("\n❌ ENHANCED TRAINING FAILED!")
        print("Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

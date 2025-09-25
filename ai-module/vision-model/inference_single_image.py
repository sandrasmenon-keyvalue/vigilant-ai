"""
Single Image Inference for Drowsiness Detection
Processes individual images and outputs drowsiness scores.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
import json

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_detection import FaceLandmarkDetector
from feature_extraction import DrowsinessFeatureExtractor
from model_training import DrowsinessModelTrainer


class SingleImageInference:
    """Inference class for single image drowsiness detection."""
    
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize inference system.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to feature scaler
        """
        # Initialize components
        self.face_detector = FaceLandmarkDetector()
        self.feature_extractor = DrowsinessFeatureExtractor()
        
        # Load trained model
        self.model_trainer = DrowsinessModelTrainer()
        self.model_trainer.load_model(model_path, scaler_path)
        
        print(f"✅ Loaded model from {model_path}")
        print(f"✅ Loaded scaler from {scaler_path}")
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process a single image and return drowsiness score.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with drowsiness score and features
        """
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect landmarks
        landmarks = self.face_detector.detect_landmarks(frame)
        
        # Extract features
        features = self.feature_extractor.extract_features(landmarks, timestamp=0.0)
        
        # Get drowsiness score
        score = self.get_drowsiness_score(features)
        
        return {
            'image_path': image_path,
            'drowsiness_score': score,
            'face_detected': features.get('face_detected', 0) > 0,
            'features': features
        }
    
    def get_drowsiness_score(self, features: Dict) -> float:
        """
        Get drowsiness score from features.
        
        Args:
            features: Extracted features
            
        Returns:
            Drowsiness score (0-1)
        """
        # Get feature names (exclude metadata columns)
        metadata_columns = {'timestamp', 'face_detected'}
        feature_names = [name for name in features.keys() if name not in metadata_columns]
        feature_names.sort()
        
        # Create feature array
        feature_array = np.array([[features.get(name, 0.0) for name in feature_names]])
        
        # Make prediction
        try:
            _, probabilities = self.model_trainer.predict(feature_array)
            score = probabilities[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            score = 0.0
        
        return score
    
    def process_image_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Process multiple images and return results.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of results
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.process_image(image_path)
                results.append(result)
                print(f"Processed {i+1}/{len(image_paths)}: {Path(image_path).name} - Score: {result['drowsiness_score']:.3f}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'drowsiness_score': 0.0,
                    'face_detected': False,
                    'error': str(e)
                })
        
        return results
    
    def process_directory(self, directory_path: str) -> List[Dict]:
        """
        Process all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            
        Returns:
            List of results
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        for img_file in directory.iterdir():
            if img_file.suffix.lower() in image_extensions:
                image_paths.append(str(img_file))
        
        if not image_paths:
            print(f"No image files found in {directory_path}")
            return []
        
        print(f"Found {len(image_paths)} images in {directory_path}")
        return self.process_image_batch(image_paths)
    
    def save_results(self, results: List[Dict], output_path: str):
        """
        Save results to JSON file.
        
        Args:
            results: List of results
            output_path: Path to save results
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✅ Results saved to {output_path}")
    
    def print_summary(self, results: List[Dict]):
        """
        Print summary of results.
        
        Args:
            results: List of results
        """
        if not results:
            print("No results to summarize")
            return
        
        total_images = len(results)
        faces_detected = sum(1 for r in results if r.get('face_detected', False))
        avg_score = np.mean([r['drowsiness_score'] for r in results])
        
        drowsy_count = sum(1 for r in results if r['drowsiness_score'] > 0.5)
        alert_count = total_images - drowsy_count
        
        print(f"\n📊 Summary:")
        print(f"  Total images: {total_images}")
        print(f"  Faces detected: {faces_detected}")
        print(f"  Average score: {avg_score:.3f}")
        print(f"  Drowsy images: {drowsy_count}")
        print(f"  Alert images: {alert_count}")


def main():
    """Main function for single image inference."""
    parser = argparse.ArgumentParser(description='Single image drowsiness detection inference')
    parser.add_argument('--model', required=True, 
                       help='Path to trained model file')
    parser.add_argument('--scaler', required=True, 
                       help='Path to feature scaler file')
    parser.add_argument('--image', 
                       help='Path to single image file')
    parser.add_argument('--directory', 
                       help='Path to directory containing images')
    parser.add_argument('--output', 
                       help='Path to save results JSON file')
    
    args = parser.parse_args()
    
    # Validate model files
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    
    if not Path(args.scaler).exists():
        print(f"❌ Scaler file not found: {args.scaler}")
        sys.exit(1)
    
    try:
        # Initialize inference system
        inference = SingleImageInference(args.model, args.scaler)
        
        if args.image:
            # Process single image
            result = inference.process_image(args.image)
            print(f"\n📸 Image: {Path(args.image).name}")
            print(f"🎯 Drowsiness Score: {result['drowsiness_score']:.3f}")
            print(f"👁️  Face Detected: {'Yes' if result['face_detected'] else 'No'}")
            print(f"📊 Status: {'DROWSY' if result['drowsiness_score'] > 0.5 else 'ALERT'}")
            
            if args.output:
                inference.save_results([result], args.output)
        
        elif args.directory:
            # Process directory
            results = inference.process_directory(args.directory)
            inference.print_summary(results)
            
            if args.output:
                inference.save_results(results, args.output)
        
        else:
            print("❌ Please provide either --image or --directory")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

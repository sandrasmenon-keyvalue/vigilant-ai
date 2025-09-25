"""
Training Script for Drowsiness Detection Model
Use this script to train a model on your real drowsiness/alert video data.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

# Add the ai-module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preparation import FrameExtractor
from face_detection import FaceLandmarkDetector
from feature_extraction import DrowsinessFeatureExtractor
from window_processing import SlidingWindowProcessor
from model_training import DrowsinessModelTrainer
from utils import visualize_features, create_detection_report


def process_dataset(dataset_dir: str, output_dir: str, fps: float = 5.0) -> pd.DataFrame:
    """
    Process a dataset of videos and extract features.
    
    Args:
        dataset_dir: Directory containing videos organized by label
        output_dir: Directory to save processed data
        fps: Frames per second to extract
        
    Returns:
        DataFrame with extracted features
    """
    print(f"Processing dataset: {dataset_dir}")
    
    # Step 1: Extract frames
    print("Step 1: Extracting frames...")
    extractor = FrameExtractor(target_fps=fps)
    frames_dir = Path(output_dir) / "frames"
    frame_df = extractor.extract_frames_from_dataset(dataset_dir, str(frames_dir))
    
    if frame_df.empty:
        raise ValueError("No frames extracted from dataset")
    
    # Step 2: Extract landmarks
    print("Step 2: Extracting facial landmarks...")
    detector = FaceLandmarkDetector()
    landmarks_data = []
    
    for idx, row in frame_df.iterrows():
        frame_path = row['frame_path']
        frame = cv2.imread(frame_path)
        
        if frame is not None:
            landmarks = detector.detect_landmarks(frame)
            if landmarks:
                landmarks['frame_path'] = frame_path
                landmarks['label'] = row['label']
                landmarks['timestamp'] = row['timestamp']
                landmarks_data.append(landmarks)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(frame_df)} frames...")
    
    print(f"Extracted landmarks from {len(landmarks_data)} frames")
    
    # Step 3: Extract features
    print("Step 3: Extracting features...")
    extractor = DrowsinessFeatureExtractor()
    features_data = []
    
    for i, landmarks in enumerate(landmarks_data):
        features = extractor.extract_features(landmarks, landmarks['timestamp'])
        features['label'] = landmarks['label']
        features['frame_path'] = landmarks['frame_path']
        features_data.append(features)
        
        if (i + 1) % 100 == 0:
            print(f"  Extracted features from {i + 1}/{len(landmarks_data)} frames...")
    
    # Step 4: Create windows
    print("Step 4: Creating feature windows...")
    processor = SlidingWindowProcessor(window_size_seconds=5.0, fps=fps)
    
    # Group features by video/session
    features_by_video = {}
    for features in features_data:
        video_source = Path(features['frame_path']).parent.name
        if video_source not in features_by_video:
            features_by_video[video_source] = []
        features_by_video[video_source].append(features)
    
    # Process each video
    all_window_features = []
    all_labels = []
    
    for video_name, video_features in features_by_video.items():
        # Sort by timestamp
        video_features.sort(key=lambda x: x['timestamp'])
        
        # Process into windows
        window_features = processor.process_feature_sequence(video_features)
        
        # Get label (should be same for all frames in video)
        label = video_features[0]['label']
        
        # Add to training data
        for window_feat in window_features:
            all_window_features.append(window_feat)
            all_labels.append(label)
    
    print(f"Created {len(all_window_features)} feature windows")
    
    # Convert to DataFrame
    if all_window_features:
        features_df = pd.DataFrame(all_window_features)
        features_df['label'] = all_labels
        
        # Save features
        features_path = Path(output_dir) / "extracted_features.csv"
        features_df.to_csv(features_path, index=False)
        print(f"Saved features to {features_path}")
        
        return features_df
    else:
        raise ValueError("No window features created")


def train_model(features_df: pd.DataFrame, output_dir: str, 
                model_type: str = 'xgboost') -> tuple:
    """
    Train a drowsiness detection model.
    
    Args:
        features_df: DataFrame with extracted features
        output_dir: Directory to save trained model
        model_type: Type of model to train
        
    Returns:
        Tuple of (model_path, scaler_path)
    """
    print(f"Training {model_type} model...")
    
    # Prepare features and labels
    feature_columns = [col for col in features_df.columns if col not in ['label', 'frame_path']]
    X = features_df[feature_columns].values
    y = (features_df['label'] == 'drowsy').astype(int).values
    
    print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Train model
    trainer = DrowsinessModelTrainer(model_type=model_type)
    results = trainer.train_model(X, y, feature_columns)
    
    # Save model
    model_path = Path(output_dir) / f"drowsiness_model_{model_type}.pkl"
    scaler_path = Path(output_dir) / f"feature_scaler_{model_type}.pkl"
    
    trainer.save_model(str(model_path), str(scaler_path))
    
    # Show feature importance
    importance = trainer.get_feature_importance()
    print("\nTop 10 most important features:")
    for i, (feature, imp) in enumerate(list(importance.items())[:10]):
        print(f"  {i+1:2d}. {feature}: {imp:.3f}")
    
    # Save feature importance
    importance_path = Path(output_dir) / f"feature_importance_{model_type}.json"
    import json
    with open(importance_path, 'w') as f:
        json.dump(importance, f, indent=2)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Feature importance saved to: {importance_path}")
    
    return str(model_path), str(scaler_path)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train drowsiness detection model')
    parser.add_argument('--dataset', required=True, 
                       help='Path to dataset directory (should contain drowsy/ and alert/ subdirectories)')
    parser.add_argument('--output', default='trained_models', 
                       help='Output directory for trained models')
    parser.add_argument('--fps', type=float, default=5.0, 
                       help='Frames per second to extract')
    parser.add_argument('--model', choices=['xgboost', 'logistic', 'random_forest'], 
                       default='xgboost', help='Model type to train')
    
    args = parser.parse_args()
    
    # Validate dataset directory
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset directory {args.dataset} does not exist")
        return
    
    # Check for label subdirectories
    drowsy_dir = dataset_path / "drowsy"
    alert_dir = dataset_path / "alert"
    
    if not drowsy_dir.exists() or not alert_dir.exists():
        print("Error: Dataset should contain 'drowsy' and 'alert' subdirectories")
        print("Expected structure:")
        print("  dataset/")
        print("    drowsy/")
        print("      video1.mp4")
        print("      video2.mp4")
        print("    alert/")
        print("      video1.mp4")
        print("      video2.mp4")
        return
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Process dataset
        features_df = process_dataset(args.dataset, str(output_path), args.fps)
        
        # Train model
        model_path, scaler_path = train_model(features_df, str(output_path), args.model)
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Models saved to: {output_path}")
        print("\nTo use the trained model:")
        print(f"  python realtime_detection.py --model {model_path} --scaler {scaler_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

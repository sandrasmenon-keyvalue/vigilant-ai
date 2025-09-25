"""
Utility functions for the drowsiness detection system.
"""

import numpy as np
import cv2
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def create_sample_images(output_dir: str, num_images: int = 20) -> None:
    """
    Create a sample dataset of images for testing the drowsiness detection system.
    
    Args:
        output_dir: Directory to save sample images
        num_images: Number of sample images to create
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "drowsy").mkdir(exist_ok=True)
    (output_path / "not_drowsy").mkdir(exist_ok=True)
    
    print(f"Creating {num_images} sample images...")
    
    for i in range(num_images):
        label = "drowsy" if i % 2 == 0 else "not_drowsy"
        image_path = output_path / label / f"sample_{i:03d}.jpg"
        
        # Create synthetic image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        if label == "drowsy":
            # Red circle for drowsy
            cv2.circle(frame, (320, 240), 50, (0, 0, 255), -1)
            cv2.putText(frame, "DROWSY", (250, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Green circle for not drowsy
            cv2.circle(frame, (320, 240), 50, (0, 255, 0), -1)
            cv2.putText(frame, "ALERT", (260, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add some text
        cv2.putText(frame, f"Sample {i}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite(str(image_path), frame)
        print(f"Created {image_path}")
    
    print(f"Sample image dataset created in {output_dir}")


def visualize_features(features_data: List[Dict], save_path: str = None) -> None:
    """
    Visualize extracted features over time.
    
    Args:
        features_data: List of feature dictionaries
        save_path: Optional path to save the plot
    """
    if not features_data:
        print("No features data to visualize")
        return
    
    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame(features_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Drowsiness Features Over Time', fontsize=16)
    
    # Plot EAR
    if 'avg_ear' in df.columns:
        axes[0, 0].plot(df['timestamp'], df['avg_ear'])
        axes[0, 0].axhline(y=0.25, color='r', linestyle='--', label='EAR Threshold')
        axes[0, 0].set_title('Eye Aspect Ratio (EAR)')
        axes[0, 0].set_ylabel('EAR')
        axes[0, 0].legend()
    
    # Plot MAR
    if 'mar' in df.columns:
        axes[0, 1].plot(df['timestamp'], df['mar'])
        axes[0, 1].axhline(y=0.5, color='r', linestyle='--', label='MAR Threshold')
        axes[0, 1].set_title('Mouth Aspect Ratio (MAR)')
        axes[0, 1].set_ylabel('MAR')
        axes[0, 1].legend()
    
    # Plot blink frequency
    if 'blink_frequency' in df.columns:
        axes[1, 0].plot(df['timestamp'], df['blink_frequency'])
        axes[1, 0].set_title('Blink Frequency')
        axes[1, 0].set_ylabel('Blinks/min')
        axes[1, 0].set_xlabel('Time (s)')
    
    # Plot head pose
    if 'head_pitch' in df.columns:
        axes[1, 1].plot(df['timestamp'], df['head_pitch'], label='Pitch')
        if 'head_yaw' in df.columns:
            axes[1, 1].plot(df['timestamp'], df['head_yaw'], label='Yaw')
        axes[1, 1].set_title('Head Pose')
        axes[1, 1].set_ylabel('Angle (degrees)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Feature visualization saved to {save_path}")
    
    plt.show()


def analyze_drowsiness_scores(scores: List[float], timestamps: List[float] = None) -> Dict:
    """
    Analyze drowsiness scores and provide statistics.
    
    Args:
        scores: List of drowsiness scores
        timestamps: Optional list of timestamps
        
    Returns:
        Dictionary with analysis results
    """
    if not scores:
        return {}
    
    scores_array = np.array(scores)
    
    analysis = {
        'total_samples': len(scores),
        'mean_score': np.mean(scores_array),
        'std_score': np.std(scores_array),
        'min_score': np.min(scores_array),
        'max_score': np.max(scores_array),
        'drowsy_percentage': np.mean(scores_array > 0.5) * 100,
        'alert_percentage': np.mean(scores_array <= 0.5) * 100,
        'high_drowsiness_percentage': np.mean(scores_array > 0.8) * 100
    }
    
    # Time-based analysis if timestamps provided
    if timestamps and len(timestamps) == len(scores):
        duration = timestamps[-1] - timestamps[0]
        analysis['duration_seconds'] = duration
        analysis['avg_score_per_minute'] = np.mean(scores_array) * 60 / duration if duration > 0 else 0
    
    return analysis


def plot_drowsiness_timeline(scores: List[float], timestamps: List[float] = None, 
                           save_path: str = None) -> None:
    """
    Plot drowsiness scores over time.
    
    Args:
        scores: List of drowsiness scores
        timestamps: Optional list of timestamps
        save_path: Optional path to save the plot
    """
    if not scores:
        print("No scores to plot")
        return
    
    if timestamps is None:
        timestamps = list(range(len(scores)))
    
    plt.figure(figsize=(12, 6))
    
    # Plot scores
    plt.plot(timestamps, scores, 'b-', alpha=0.7, label='Drowsiness Score')
    
    # Add threshold line
    plt.axhline(y=0.5, color='r', linestyle='--', label='Drowsiness Threshold')
    
    # Fill areas
    plt.fill_between(timestamps, 0, 0.5, alpha=0.2, color='green', label='Alert Zone')
    plt.fill_between(timestamps, 0.5, 1.0, alpha=0.2, color='red', label='Drowsy Zone')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Drowsiness Score')
    plt.title('Drowsiness Detection Timeline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Drowsiness timeline saved to {save_path}")
    
    plt.show()


def save_detection_results(scores: List[float], features: List[Dict], 
                          output_path: str) -> None:
    """
    Save detection results to JSON file.
    
    Args:
        scores: List of drowsiness scores
        features: List of feature dictionaries
        output_path: Path to save results
    """
    results = {
        'drowsiness_scores': scores,
        'features': features,
        'analysis': analyze_drowsiness_scores(scores),
        'metadata': {
            'total_frames': len(scores),
            'timestamp': str(pd.Timestamp.now())
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Detection results saved to {output_path}")


def load_detection_results(input_path: str) -> Dict:
    """
    Load detection results from JSON file.
    
    Args:
        input_path: Path to results file
        
    Returns:
        Dictionary with loaded results
    """
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    return results


def benchmark_image_processing(image_paths: List[str]) -> Dict:
    """
    Benchmark the detection speed on images.
    
    Args:
        image_paths: List of image paths to process
        
    Returns:
        Dictionary with timing results
    """
    import time
    from .face_detection import FaceLandmarkDetector
    from .feature_extraction import DrowsinessFeatureExtractor
    
    # Initialize components
    face_detector = FaceLandmarkDetector()
    feature_extractor = DrowsinessFeatureExtractor()
    
    total_time = 0
    processed_count = 0
    
    print(f"Benchmarking detection speed on {len(image_paths)} images...")
    
    for i, image_path in enumerate(image_paths):
        frame = cv2.imread(image_path)
        if frame is None:
            continue
        
        start_time = time.time()
        
        # Process image
        landmarks = face_detector.detect_landmarks(frame)
        features = feature_extractor.extract_features(landmarks, 0.0)
        
        end_time = time.time()
        total_time += (end_time - start_time)
        processed_count += 1
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} images...")
    
    avg_time_per_image = total_time / processed_count if processed_count > 0 else 0
    
    results = {
        'images_processed': processed_count,
        'total_time': total_time,
        'avg_time_per_image': avg_time_per_image,
        'images_per_second': 1.0 / avg_time_per_image if avg_time_per_image > 0 else 0
    }
    
    print(f"Benchmark Results:")
    print(f"  Images processed: {processed_count}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per image: {avg_time_per_image:.3f}s")
    print(f"  Images per second: {results['images_per_second']:.1f}")
    
    return results


def create_detection_report(image_paths: List[str], scores: List[float], 
                          output_dir: str) -> None:
    """
    Create a comprehensive detection report.
    
    Args:
        image_paths: List of analyzed image paths
        scores: List of drowsiness scores
        output_dir: Directory to save report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Analyze scores
    analysis = analyze_drowsiness_scores(scores)
    
    # Create summary report
    report_path = output_path / "detection_report.txt"
    with open(report_path, 'w') as f:
        f.write("DROWSINESS DETECTION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Images Analyzed: {len(image_paths)}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Images: {analysis['total_samples']}\n")
        f.write(f"Mean Score: {analysis['mean_score']:.3f}\n")
        f.write(f"Std Deviation: {analysis['std_score']:.3f}\n")
        f.write(f"Min Score: {analysis['min_score']:.3f}\n")
        f.write(f"Max Score: {analysis['max_score']:.3f}\n\n")
        
        f.write("DROWSINESS LEVELS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Not Drowsy (≤0.5): {analysis['alert_percentage']:.1f}%\n")
        f.write(f"Drowsy (>0.5): {analysis['drowsy_percentage']:.1f}%\n")
        f.write(f"High Drowsiness (>0.8): {analysis['high_drowsiness_percentage']:.1f}%\n\n")
    
    print(f"Detection report saved to {output_dir}")


def main():
    """Demo utility functions."""
    print("Utility functions demo")
    
    # Create sample dataset
    create_sample_images("sample_data", num_images=10)
    
    # Create sample scores for visualization
    np.random.seed(42)
    scores = np.random.uniform(0, 1, 50)  # Random scores for 50 images
    
    # Analyze scores
    analysis = analyze_drowsiness_scores(scores)
    print(f"Analysis: {analysis}")
    
    print("Utility functions demo completed!")


if __name__ == "__main__":
    main()

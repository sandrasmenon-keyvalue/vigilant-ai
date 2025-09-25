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


def create_sample_dataset(output_dir: str, num_videos: int = 20) -> None:
    """
    Create a sample dataset for testing the drowsiness detection system.
    
    Args:
        output_dir: Directory to save sample videos
        num_videos: Number of sample videos to create
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "drowsy").mkdir(exist_ok=True)
    (output_path / "alert").mkdir(exist_ok=True)
    
    print(f"Creating {num_videos} sample videos...")
    
    for i in range(num_videos):
        label = "drowsy" if i % 2 == 0 else "alert"
        video_path = output_path / label / f"sample_{i:03d}.mp4"
        
        # Create synthetic video with different patterns for drowsy vs alert
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
        
        for frame_num in range(300):  # 10 seconds at 30 FPS
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            if label == "drowsy":
                # Slower, more erratic movement (simulating drowsy behavior)
                x = int(320 + 80 * np.sin(frame_num * 0.03))
                y = int(240 + 40 * np.cos(frame_num * 0.02))
                color = (0, 0, 255)  # Red for drowsy
            else:
                # Faster, more regular movement (simulating alert behavior)
                x = int(320 + 120 * np.sin(frame_num * 0.08))
                y = int(240 + 80 * np.cos(frame_num * 0.06))
                color = (0, 255, 0)  # Green for alert
            
            # Draw moving circle
            cv2.circle(frame, (x, y), 25, color, -1)
            
            # Add some text
            cv2.putText(frame, f"Sample {i} - {label}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"Created {video_path}")
    
    print(f"Sample dataset created in {output_dir}")


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


def benchmark_detection_speed(video_path: str, num_frames: int = 100) -> Dict:
    """
    Benchmark the detection speed on a video.
    
    Args:
        video_path: Path to test video
        num_frames: Number of frames to process
        
    Returns:
        Dictionary with timing results
    """
    import time
    from .face_detection import FaceLandmarkDetector
    from .feature_extraction import DrowsinessFeatureExtractor
    
    # Initialize components
    face_detector = FaceLandmarkDetector()
    feature_extractor = DrowsinessFeatureExtractor()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_count = 0
    total_time = 0
    
    print(f"Benchmarking detection speed on {num_frames} frames...")
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Process frame
        landmarks = face_detector.detect_landmarks(frame)
        features = feature_extractor.extract_features(landmarks, frame_count * 0.2)
        
        end_time = time.time()
        total_time += (end_time - start_time)
        frame_count += 1
        
        if frame_count % 20 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    
    avg_time_per_frame = total_time / frame_count
    fps = 1.0 / avg_time_per_frame
    
    results = {
        'frames_processed': frame_count,
        'total_time': total_time,
        'avg_time_per_frame': avg_time_per_frame,
        'estimated_fps': fps,
        'can_realtime': fps >= 5.0  # Can we achieve 5 FPS?
    }
    
    print(f"Benchmark Results:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per frame: {avg_time_per_frame:.3f}s")
    print(f"  Estimated FPS: {fps:.1f}")
    print(f"  Can achieve real-time (5 FPS): {results['can_realtime']}")
    
    return results


def create_detection_report(video_path: str, scores: List[float], 
                          output_dir: str) -> None:
    """
    Create a comprehensive detection report.
    
    Args:
        video_path: Path to analyzed video
        scores: List of drowsiness scores
        output_dir: Directory to save report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Analyze scores
    analysis = analyze_drowsiness_scores(scores)
    
    # Create timeline plot
    timeline_path = output_path / "drowsiness_timeline.png"
    plot_drowsiness_timeline(scores, save_path=str(timeline_path))
    
    # Create summary report
    report_path = output_path / "detection_report.txt"
    with open(report_path, 'w') as f:
        f.write("DROWSINESS DETECTION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Samples: {analysis['total_samples']}\n")
        f.write(f"Mean Score: {analysis['mean_score']:.3f}\n")
        f.write(f"Std Deviation: {analysis['std_score']:.3f}\n")
        f.write(f"Min Score: {analysis['min_score']:.3f}\n")
        f.write(f"Max Score: {analysis['max_score']:.3f}\n\n")
        
        f.write("DROWSINESS LEVELS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Alert (≤0.5): {analysis['alert_percentage']:.1f}%\n")
        f.write(f"Drowsy (>0.5): {analysis['drowsy_percentage']:.1f}%\n")
        f.write(f"High Drowsiness (>0.8): {analysis['high_drowsiness_percentage']:.1f}%\n\n")
        
        if 'duration_seconds' in analysis:
            f.write(f"Duration: {analysis['duration_seconds']:.1f} seconds\n")
    
    print(f"Detection report saved to {output_dir}")


def main():
    """Demo utility functions."""
    print("Utility functions demo")
    
    # Create sample dataset
    create_sample_dataset("sample_data", num_videos=10)
    
    # Create sample scores for visualization
    np.random.seed(42)
    timestamps = np.linspace(0, 60, 300)  # 5 minutes at 5 FPS
    scores = 0.3 + 0.4 * np.sin(timestamps * 0.1) + 0.1 * np.random.randn(300)
    scores = np.clip(scores, 0, 1)
    
    # Analyze scores
    analysis = analyze_drowsiness_scores(scores, timestamps)
    print(f"Analysis: {analysis}")
    
    # Plot timeline
    plot_drowsiness_timeline(scores, timestamps)
    
    print("Utility functions demo completed!")


if __name__ == "__main__":
    main()

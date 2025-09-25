"""
Step 1: Data Preparation
Extract frames at 5 FPS from video datasets for drowsiness detection.
"""

import cv2
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd


class FrameExtractor:
    """Extract frames from videos at specified FPS."""
    
    def __init__(self, target_fps: int = 5):
        """
        Initialize frame extractor.
        
        Args:
            target_fps: Frames per second to extract (default: 5)
        """
        self.target_fps = target_fps
        
    def extract_frames_from_video(self, video_path: str, output_dir: str, 
                                 label: str = "unknown") -> List[str]:
        """
        Extract frames from a single video file.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted frames
            label: Label for the frames (e.g., 'drowsy', 'alert')
            
        Returns:
            List of paths to extracted frame files
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps
        
        # Calculate frame interval for target FPS
        frame_interval = int(original_fps / self.target_fps)
        
        # Create output directory
        video_name = Path(video_path).stem
        frame_dir = Path(output_dir) / f"{video_name}_{label}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_frames = []
        frame_count = 0
        saved_count = 0
        
        print(f"Extracting frames from {video_path}")
        print(f"Original FPS: {original_fps:.2f}, Target FPS: {self.target_fps}")
        print(f"Duration: {duration:.2f}s, Expected frames: {int(duration * self.target_fps)}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save frame at specified intervals
            if frame_count % frame_interval == 0:
                frame_filename = f"frame_{saved_count:06d}.jpg"
                frame_path = frame_dir / frame_filename
                
                cv2.imwrite(str(frame_path), frame)
                extracted_frames.append(str(frame_path))
                saved_count += 1
                
            frame_count += 1
            
        cap.release()
        print(f"Extracted {saved_count} frames to {frame_dir}")
        
        return extracted_frames
    
    def extract_frames_from_dataset(self, dataset_dir: str, output_dir: str) -> pd.DataFrame:
        """
        Extract frames from entire dataset directory.
        
        Args:
            dataset_dir: Directory containing video files
            output_dir: Directory to save extracted frames
            
        Returns:
            DataFrame with frame paths and labels
        """
        dataset_path = Path(dataset_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Supported video formats
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        
        frame_data = []
        
        # Process each video file
        for video_file in dataset_path.rglob('*'):
            if video_file.suffix.lower() in video_extensions:
                # Determine label from directory structure
                # Expected: dataset_dir/label/video_file
                relative_path = video_file.relative_to(dataset_path)
                label = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
                
                try:
                    extracted_frames = self.extract_frames_from_video(
                        str(video_file), str(output_path), label
                    )
                    
                    # Add to frame data
                    for frame_path in extracted_frames:
                        frame_data.append({
                            'frame_path': frame_path,
                            'video_source': str(video_file),
                            'label': label,
                            'timestamp': self._get_timestamp_from_path(frame_path)
                        })
                        
                except Exception as e:
                    print(f"Error processing {video_file}: {e}")
                    continue
        
        # Create DataFrame
        df = pd.DataFrame(frame_data)
        
        if not df.empty:
            # Save frame metadata
            metadata_path = output_path / "frame_metadata.csv"
            df.to_csv(metadata_path, index=False)
            print(f"Saved frame metadata to {metadata_path}")
            print(f"Total frames extracted: {len(df)}")
            print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def _get_timestamp_from_path(self, frame_path: str) -> float:
        """Extract timestamp from frame filename."""
        frame_name = Path(frame_path).stem
        # Assuming format: frame_XXXXXX.jpg
        try:
            frame_number = int(frame_name.split('_')[1])
            return frame_number / self.target_fps
        except:
            return 0.0
    
    def create_sample_dataset(self, output_dir: str, num_videos: int = 10):
        """
        Create a sample dataset for testing (synthetic videos).
        
        Args:
            output_dir: Directory to save sample videos
            num_videos: Number of sample videos to create
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different labels
        (output_path / "drowsy").mkdir(exist_ok=True)
        (output_path / "alert").mkdir(exist_ok=True)
        
        print(f"Creating {num_videos} sample videos...")
        
        for i in range(num_videos):
            # Create synthetic video with moving shapes
            label = "drowsy" if i % 2 == 0 else "alert"
            video_path = output_path / label / f"sample_{i:03d}.mp4"
            
            # Create a simple synthetic video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
            
            for frame_num in range(300):  # 10 seconds at 30 FPS
                # Create frame with moving circle
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Different movement patterns for drowsy vs alert
                if label == "drowsy":
                    # Slower, more erratic movement
                    x = int(320 + 100 * np.sin(frame_num * 0.05))
                    y = int(240 + 50 * np.cos(frame_num * 0.03))
                else:
                    # Faster, more regular movement
                    x = int(320 + 150 * np.sin(frame_num * 0.1))
                    y = int(240 + 100 * np.cos(frame_num * 0.08))
                
                cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)
                out.write(frame)
            
            out.release()
            print(f"Created {video_path}")


def main():
    """Demo the frame extraction process."""
    extractor = FrameExtractor(target_fps=5)
    
    # Create sample dataset for testing
    sample_dir = "sample_dataset"
    extractor.create_sample_dataset(sample_dir, num_videos=6)
    
    # Extract frames from sample dataset
    output_dir = "extracted_frames"
    frame_df = extractor.extract_frames_from_dataset(sample_dir, output_dir)
    
    print("\nFrame extraction completed!")
    print(f"Extracted {len(frame_df)} frames")
    if not frame_df.empty:
        print("\nSample data:")
        print(frame_df.head())


if __name__ == "__main__":
    main()

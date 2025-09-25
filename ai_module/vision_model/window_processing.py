"""
Step 4: Window Processing
Create 5-second sliding windows with feature aggregation for drowsiness detection.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import deque
import json


class SlidingWindowProcessor:
    """Process features in sliding windows for drowsiness detection."""
    
    def __init__(self, window_size_seconds: float = 5.0, fps: float = 5.0):
        """
        Initialize sliding window processor.
        
        Args:
            window_size_seconds: Size of sliding window in seconds
            fps: Frames per second
        """
        self.window_size_seconds = window_size_seconds
        self.fps = fps
        self.window_size_frames = int(window_size_seconds * fps)
        
        # Buffer to store recent features
        self.feature_buffer = deque(maxlen=self.window_size_frames)
        
        # Statistics for feature normalization
        self.feature_stats = {}
        
    def add_feature(self, features: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        Add a new feature vector to the buffer and return window features if window is full.
        
        Args:
            features: Feature dictionary from feature extraction
            
        Returns:
            Aggregated window features or None if window not full
        """
        self.feature_buffer.append(features)
        
        # Only return window features when buffer is full
        if len(self.feature_buffer) < self.window_size_frames:
            return None
        
        return self.aggregate_window_features()
    
    def aggregate_window_features(self) -> Dict[str, float]:
        """
        Aggregate features over the current window.
        
        Returns:
            Dictionary with aggregated features
        """
        if len(self.feature_buffer) == 0:
            return {}
        
        # Convert buffer to DataFrame for easier processing
        df = pd.DataFrame(list(self.feature_buffer))
        
        # Calculate aggregated features
        window_features = {}
        
        # Basic statistical aggregations
        for column in df.columns:
            if column == 'timestamp':
                continue
                
            values = df[column].values
            
            # Mean
            window_features[f'{column}_mean'] = np.mean(values)
            
            # Standard deviation
            window_features[f'{column}_std'] = np.std(values)
            
            # Min and max
            window_features[f'{column}_min'] = np.min(values)
            window_features[f'{column}_max'] = np.max(values)
            
            # For binary features, use sum (count of occurrences)
            if column in ['blink_detected', 'nod_detected', 'yawn_indicator']:
                window_features[f'{column}_count'] = np.sum(values)
                window_features[f'{column}_rate'] = np.sum(values) / len(values)
        
        # Special aggregations for drowsiness-specific features
        
        # Eye closure patterns
        ear_values = df['avg_ear'].values
        window_features['ear_trend'] = self._calculate_trend(ear_values)
        window_features['ear_volatility'] = np.std(ear_values)
        window_features['low_ear_ratio'] = np.sum(ear_values < 0.25) / len(ear_values)
        
        # Blink patterns
        if 'blink_detected' in df.columns:
            blink_times = df[df['blink_detected'] == 1.0]['timestamp'].values
            if len(blink_times) > 1:
                blink_intervals = np.diff(blink_times)
                window_features['avg_blink_interval'] = np.mean(blink_intervals)
                window_features['blink_regularity'] = 1.0 / (np.std(blink_intervals) + 1e-6)
            else:
                window_features['avg_blink_interval'] = self.window_size_seconds
                window_features['blink_regularity'] = 0.0
        
        # Head movement patterns
        if 'head_pitch' in df.columns:
            pitch_values = df['head_pitch'].values
            window_features['head_movement_amplitude'] = np.max(pitch_values) - np.min(pitch_values)
            window_features['head_movement_frequency'] = self._count_peaks(pitch_values)
        
        # Yawn patterns
        if 'mar' in df.columns:
            mar_values = df['mar'].values
            window_features['yawn_duration'] = self._calculate_yawn_duration(mar_values)
            window_features['max_mar'] = np.max(mar_values)
        
        # Face detection reliability
        if 'face_detected' in df.columns:
            face_detection_rate = np.mean(df['face_detected'].values)
            window_features['face_detection_reliability'] = face_detection_rate
        
        # Temporal features
        if 'timestamp' in df.columns:
            timestamps = df['timestamp'].values
            window_features['window_duration'] = timestamps[-1] - timestamps[0]
            window_features['frame_count'] = len(df)
        
        return window_features
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate linear trend of values over time."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope
    
    def _count_peaks(self, values: np.ndarray, threshold: float = 0.1) -> int:
        """Count peaks in a signal."""
        if len(values) < 3:
            return 0
        
        peaks = 0
        for i in range(1, len(values) - 1):
            if (values[i] > values[i-1] and 
                values[i] > values[i+1] and 
                values[i] > threshold):
                peaks += 1
        
        return peaks
    
    def _calculate_yawn_duration(self, mar_values: np.ndarray, threshold: float = 0.5) -> float:
        """Calculate total duration of yawns in the window."""
        yawn_frames = mar_values > threshold
        if not np.any(yawn_frames):
            return 0.0
        
        # Find continuous yawn segments
        yawn_segments = []
        in_yawn = False
        start_frame = 0
        
        for i, is_yawn in enumerate(yawn_frames):
            if is_yawn and not in_yawn:
                start_frame = i
                in_yawn = True
            elif not is_yawn and in_yawn:
                yawn_segments.append(i - start_frame)
                in_yawn = False
        
        # Handle case where yawn continues to end of window
        if in_yawn:
            yawn_segments.append(len(yawn_frames) - start_frame)
        
        # Convert frame count to seconds
        total_yawn_frames = sum(yawn_segments)
        return total_yawn_frames / self.fps
    
    def process_feature_sequence(self, feature_sequence: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Process a sequence of features and return window features.
        
        Args:
            feature_sequence: List of feature dictionaries
            
        Returns:
            List of window feature dictionaries
        """
        window_features = []
        
        for features in feature_sequence:
            window_feat = self.add_feature(features)
            if window_feat is not None:
                window_features.append(window_feat)
        
        return window_features
    
    def create_training_data(self, feature_sequences: List[List[Dict]], 
                           labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training data from feature sequences and labels.
        
        Args:
            feature_sequences: List of feature sequences (one per video/session)
            labels: List of labels ('drowsy' or 'alert')
            
        Returns:
            Tuple of (features_array, labels_array)
        """
        all_window_features = []
        all_labels = []
        
        for features_seq, label in zip(feature_sequences, labels):
            # Process this sequence
            window_features = self.process_feature_sequence(features_seq)
            
            # Add to training data
            for window_feat in window_features:
                all_window_features.append(window_feat)
                all_labels.append(1.0 if label == 'drowsy' else 0.0)
        
        # Convert to arrays
        if not all_window_features:
            return np.array([]), np.array([])
        
        # Get all feature names
        feature_names = list(all_window_features[0].keys())
        feature_names.sort()  # Ensure consistent ordering
        
        # Create feature matrix
        X = np.array([[feat[name] for name in feature_names] for feat in all_window_features])
        y = np.array(all_labels)
        
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Get list of all window feature names."""
        # This would be populated after processing some data
        # For now, return expected feature names
        base_features = [
            'avg_ear', 'mar', 'head_pitch', 'head_yaw', 'head_roll',
            'blink_detected', 'nod_detected', 'yawn_indicator',
            'blink_frequency', 'nod_frequency', 'eye_closure', 'face_detected'
        ]
        
        feature_names = []
        for base_feat in base_features:
            feature_names.extend([
                f'{base_feat}_mean', f'{base_feat}_std', 
                f'{base_feat}_min', f'{base_feat}_max'
            ])
            
            if base_feat in ['blink_detected', 'nod_detected', 'yawn_indicator']:
                feature_names.extend([f'{base_feat}_count', f'{base_feat}_rate'])
        
        # Add special features
        feature_names.extend([
            'ear_trend', 'ear_volatility', 'low_ear_ratio',
            'avg_blink_interval', 'blink_regularity',
            'head_movement_amplitude', 'head_movement_frequency',
            'yawn_duration', 'max_mar', 'face_detection_reliability',
            'window_duration', 'frame_count'
        ])
        
        return sorted(feature_names)
    
    def reset(self):
        """Reset the feature buffer."""
        self.feature_buffer.clear()
    
    def save_window_features(self, window_features: List[Dict], output_path: str):
        """Save window features to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(window_features, f, indent=2)
        print(f"Saved {len(window_features)} window features to {output_path}")


def main():
    """Demo the window processing."""
    processor = SlidingWindowProcessor(window_size_seconds=5.0, fps=5.0)
    
    # Create sample feature sequence
    sample_features = []
    for i in range(30):  # 6 seconds of data at 5 FPS
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
        sample_features.append(features)
    
    # Process features
    window_features = processor.process_feature_sequence(sample_features)
    
    print(f"Processed {len(sample_features)} frames into {len(window_features)} windows")
    
    if window_features:
        print(f"Window feature names: {len(processor.get_feature_names())}")
        print("Sample window features:")
        for name, value in list(window_features[0].items())[:10]:
            print(f"  {name}: {value:.3f}")
    
    print("Window processing module ready!")


if __name__ == "__main__":
    main()

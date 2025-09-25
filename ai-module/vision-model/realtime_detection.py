"""
Step 6 & 7: Real-time Detection and Score Smoothing
Real-time drowsiness detection pipeline with temporal smoothing.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from collections import deque
import threading
import queue

from .face_detection import FaceLandmarkDetector
from .feature_extraction import DrowsinessFeatureExtractor
from .window_processing import SlidingWindowProcessor
from .model_training import DrowsinessModelTrainer
import joblib


class RealTimeDrowsinessDetector:
    """Real-time drowsiness detection system."""
    
    def __init__(self, model_path: str, scaler_path: str, 
                 fps: float = 5.0, window_size_seconds: float = 5.0):
        """
        Initialize real-time drowsiness detector.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to feature scaler
            fps: Target frames per second
            window_size_seconds: Size of sliding window
        """
        self.fps = fps
        self.frame_interval = 1.0 / fps
        
        # Initialize components
        self.face_detector = FaceLandmarkDetector()
        self.feature_extractor = DrowsinessFeatureExtractor()
        self.window_processor = SlidingWindowProcessor(
            window_size_seconds=window_size_seconds, 
            fps=fps
        )
        
        # Load trained model
        self.model_trainer = DrowsinessModelTrainer()
        self.model_trainer.load_model(model_path, scaler_path)
        
        # Score smoothing
        self.score_buffer = deque(maxlen=10)  # Last 10 scores
        self.smoothing_alpha = 0.3  # Exponential smoothing factor
        self.smoothed_score = 0.0
        
        # State tracking
        self.is_running = False
        self.current_score = 0.0
        self.last_prediction_time = 0.0
        
        # Threading for real-time processing
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Process a single frame and return drowsiness features.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with extracted features
        """
        # Detect landmarks
        landmarks = self.face_detector.detect_landmarks(frame)
        
        # Extract features
        current_time = time.time()
        features = self.feature_extractor.extract_features(landmarks, current_time)
        
        return features
    
    def get_drowsiness_score(self, features: Dict[str, float]) -> float:
        """
        Get drowsiness score from features.
        
        Args:
            features: Extracted features
            
        Returns:
            Drowsiness score (0-1)
        """
        # Add features to window processor
        window_features = self.window_processor.add_feature(features)
        
        if window_features is None:
            # Window not full yet, return current smoothed score
            return self.smoothed_score
        
        # Convert window features to array
        feature_names = self.window_processor.get_feature_names()
        feature_array = np.array([[window_features.get(name, 0.0) for name in feature_names]])
        
        # Make prediction
        try:
            _, probabilities = self.model_trainer.predict(feature_array)
            raw_score = probabilities[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            raw_score = 0.0
        
        # Apply temporal smoothing
        self.score_buffer.append(raw_score)
        self.smoothed_score = self._apply_smoothing(raw_score)
        
        return self.smoothed_score
    
    def _apply_smoothing(self, new_score: float) -> float:
        """
        Apply temporal smoothing to the drowsiness score.
        
        Args:
            new_score: New raw score
            
        Returns:
            Smoothed score
        """
        # Exponential smoothing
        if self.smoothed_score == 0.0:
            self.smoothed_score = new_score
        else:
            self.smoothed_score = (self.smoothing_alpha * new_score + 
                                 (1 - self.smoothing_alpha) * self.smoothed_score)
        
        return self.smoothed_score
    
    def process_video_stream(self, video_source: int = 0, 
                           display: bool = True) -> None:
        """
        Process video stream in real-time.
        
        Args:
            video_source: Video source (0 for webcam, or video file path)
            display: Whether to display the video
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Camera FPS
        
        self.is_running = True
        last_frame_time = 0.0
        
        print("Starting real-time drowsiness detection...")
        print("Press 'q' to quit, 'r' to reset counters")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                
                # Control frame rate
                if current_time - last_frame_time < self.frame_interval:
                    continue
                
                last_frame_time = current_time
                
                # Process frame
                features = self.process_frame(frame)
                drowsiness_score = self.get_drowsiness_score(features)
                
                # Update display
                if display:
                    annotated_frame = self._annotate_frame(frame, features, drowsiness_score)
                    cv2.imshow('Drowsiness Detection', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        self._reset_detector()
                
                # Print score every second
                if int(current_time) != int(self.last_prediction_time):
                    self._print_status(drowsiness_score, features)
                    self.last_prediction_time = current_time
                
        except KeyboardInterrupt:
            print("\nStopping detection...")
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            self.is_running = False
    
    def _annotate_frame(self, frame: np.ndarray, features: Dict[str, float], 
                       score: float) -> np.ndarray:
        """
        Annotate frame with detection results.
        
        Args:
            frame: Input frame
            features: Extracted features
            score: Drowsiness score
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw face landmarks if available
        if features.get('face_detected', 0) > 0:
            # This would require the landmark data, simplified for now
            pass
        
        # Draw drowsiness score
        score_text = f"Drowsiness: {score:.2f}"
        color = (0, 255, 0) if score < 0.5 else (0, 0, 255)  # Green if alert, red if drowsy
        
        cv2.putText(annotated, score_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw status bar
        bar_width = int(w * score)
        cv2.rectangle(annotated, (0, h-20), (bar_width, h), color, -1)
        cv2.rectangle(annotated, (0, h-20), (w, h), (255, 255, 255), 2)
        
        # Draw feature indicators
        y_offset = 60
        key_features = ['avg_ear', 'mar', 'blink_frequency', 'eye_closure']
        
        for feature in key_features:
            if feature in features:
                value = features[feature]
                text = f"{feature}: {value:.3f}"
                cv2.putText(annotated, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
        
        return annotated
    
    def _print_status(self, score: float, features: Dict[str, float]):
        """Print current status."""
        status = "DROWSY" if score > 0.5 else "ALERT"
        print(f"[{time.strftime('%H:%M:%S')}] {status} - Score: {score:.3f} | "
              f"EAR: {features.get('avg_ear', 0):.3f} | "
              f"MAR: {features.get('mar', 0):.3f} | "
              f"Blinks: {features.get('blink_frequency', 0):.1f}/min")
    
    def _reset_detector(self):
        """Reset detector state."""
        self.feature_extractor.reset_counters()
        self.window_processor.reset()
        self.score_buffer.clear()
        self.smoothed_score = 0.0
        print("Detector reset!")
    
    def process_video_file(self, video_path: str, output_path: str = None) -> List[float]:
        """
        Process a video file and return drowsiness scores.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
            
        Returns:
            List of drowsiness scores
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / self.fps)
        
        scores = []
        frame_count = 0
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, (640, 480))
        
        print(f"Processing video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame
            if frame_count % frame_interval == 0:
                features = self.process_frame(frame)
                score = self.get_drowsiness_score(features)
                scores.append(score)
                
                if writer:
                    annotated_frame = self._annotate_frame(frame, features, score)
                    writer.write(cv2.resize(annotated_frame, (640, 480)))
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        if writer:
            writer.release()
        
        print(f"Processed {frame_count} frames, generated {len(scores)} scores")
        return scores
    
    def get_current_status(self) -> Dict[str, float]:
        """
        Get current detection status.
        
        Returns:
            Dictionary with current status
        """
        return {
            'drowsiness_score': self.smoothed_score,
            'raw_score': self.score_buffer[-1] if self.score_buffer else 0.0,
            'is_drowsy': self.smoothed_score > 0.5,
            'confidence': abs(self.smoothed_score - 0.5) * 2,  # Distance from threshold
            'buffer_size': len(self.score_buffer)
        }


def main():
    """Demo the real-time detection system."""
    print("Real-time Drowsiness Detection Demo")
    print("Note: This demo requires a trained model. Run model_training.py first.")
    
    # For demo purposes, create dummy model files
    # In real usage, these would be created by training
    model_path = "drowsiness_model.pkl"
    scaler_path = "feature_scaler.pkl"
    
    try:
        # Initialize detector
        detector = RealTimeDrowsinessDetector(model_path, scaler_path)
        
        # Start real-time detection
        detector.process_video_stream(video_source=0, display=True)
        
    except FileNotFoundError:
        print("Model files not found. Please train a model first using model_training.py")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

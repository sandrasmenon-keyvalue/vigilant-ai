"""
Step 2: Face Detection and Landmark Extraction
Use MediaPipe to detect faces and extract facial landmarks for drowsiness analysis.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict
import json
from pathlib import Path


class FaceLandmarkDetector:
    """Detect faces and extract landmarks using MediaPipe."""
    
    def __init__(self, min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe face detection.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize face mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,  # Focus on single face for driver
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Define important landmark indices for drowsiness detection
        self.landmark_indices = {
            # Left eye landmarks (outer to inner)
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            # Right eye landmarks (outer to inner)  
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            # Mouth landmarks
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            # Nose landmarks for head pose
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 360, 279],
            # Face outline for head pose
            'face_outline': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        }
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect face landmarks in an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary containing landmark data or None if no face detected
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get the first (and only) face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract landmark coordinates
        landmarks = []
        h, w = image.shape[:2]
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            z = landmark.z  # Relative depth
            landmarks.append((x, y, z))
        
        # Extract specific landmark groups
        landmark_data = {
            'all_landmarks': landmarks,
            'left_eye': [landmarks[i] for i in self.landmark_indices['left_eye']],
            'right_eye': [landmarks[i] for i in self.landmark_indices['right_eye']],
            'mouth': [landmarks[i] for i in self.landmark_indices['mouth']],
            'nose': [landmarks[i] for i in self.landmark_indices['nose']],
            'face_outline': [landmarks[i] for i in self.landmark_indices['face_outline']],
            'image_shape': (h, w)
        }
        
        return landmark_data
    
    def draw_landmarks(self, image: np.ndarray, landmark_data: Dict, 
                      draw_all: bool = False) -> np.ndarray:
        """
        Draw landmarks on the image for visualization.
        
        Args:
            image: Input image
            landmark_data: Landmark data from detect_landmarks
            draw_all: Whether to draw all landmarks or just key ones
            
        Returns:
            Image with landmarks drawn
        """
        if landmark_data is None:
            return image
        
        image_copy = image.copy()
        
        if draw_all:
            # Draw all landmarks
            for landmark in landmark_data['all_landmarks']:
                cv2.circle(image_copy, (landmark[0], landmark[1]), 1, (0, 255, 0), -1)
        else:
            # Draw key landmark groups with different colors
            colors = {
                'left_eye': (255, 0, 0),    # Blue
                'right_eye': (0, 255, 0),   # Green  
                'mouth': (0, 0, 255),       # Red
                'nose': (255, 255, 0),      # Cyan
            }
            
            for group_name, color in colors.items():
                if group_name in landmark_data:
                    for landmark in landmark_data[group_name]:
                        cv2.circle(image_copy, (landmark[0], landmark[1]), 2, color, -1)
        
        return image_copy
    
    def get_face_bbox(self, landmark_data: Dict) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box of the detected face.
        
        Args:
            landmark_data: Landmark data from detect_landmarks
            
        Returns:
            Tuple of (x, y, width, height) or None
        """
        if landmark_data is None:
            return None
        
        landmarks = landmark_data['all_landmarks']
        if not landmarks:
            return None
        
        # Get min/max coordinates
        xs = [landmark[0] for landmark in landmarks]
        ys = [landmark[1] for landmark in landmarks]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Add some padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(landmark_data['image_shape'][1], x_max + padding)
        y_max = min(landmark_data['image_shape'][0], y_max + padding)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def process_video_frames(self, video_path: str, output_dir: str = None) -> List[Dict]:
        """
        Process all frames in a video and extract landmarks.
        
        Args:
            video_path: Path to input video
            output_dir: Optional directory to save processed frames
            
        Returns:
            List of landmark data for each frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_landmarks = []
        frame_count = 0
        
        print(f"Processing video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect landmarks
            landmark_data = self.detect_landmarks(frame)
            
            if landmark_data:
                landmark_data['frame_number'] = frame_count
                landmark_data['timestamp'] = frame_count / cap.get(cv2.CAP_PROP_FPS)
                frame_landmarks.append(landmark_data)
                
                # Save processed frame if output directory specified
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    # Draw landmarks on frame
                    annotated_frame = self.draw_landmarks(frame, landmark_data)
                    
                    # Save frame
                    frame_filename = f"frame_{frame_count:06d}_landmarks.jpg"
                    cv2.imwrite(str(output_path / frame_filename), annotated_frame)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        print(f"Processed {frame_count} frames, detected landmarks in {len(frame_landmarks)} frames")
        
        return frame_landmarks
    
    def save_landmarks(self, landmarks_data: List[Dict], output_path: str):
        """
        Save landmark data to JSON file.
        
        Args:
            landmarks_data: List of landmark data
            output_path: Path to save JSON file
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = []
        
        for frame_data in landmarks_data:
            serializable_frame = {}
            for key, value in frame_data.items():
                if key == 'all_landmarks':
                    serializable_frame[key] = [[float(x), float(y), float(z)] for x, y, z in value]
                elif key in ['left_eye', 'right_eye', 'mouth', 'nose', 'face_outline']:
                    serializable_frame[key] = [[float(x), float(y), float(z)] for x, y, z in value]
                else:
                    serializable_frame[key] = value
            serializable_data.append(serializable_frame)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"Saved landmark data to {output_path}")


def main():
    """Demo the face landmark detection."""
    detector = FaceLandmarkDetector()
    
    # Test with webcam or sample image
    print("Testing face landmark detection...")
    
    # Create a simple test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "Place your face here", (200, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Try to detect landmarks (will return None for synthetic image)
    landmarks = detector.detect_landmarks(test_image)
    
    if landmarks:
        print("Face detected!")
        print(f"Found {len(landmarks['all_landmarks'])} landmarks")
    else:
        print("No face detected in test image (expected for synthetic image)")
    
    print("Face landmark detection module ready!")


if __name__ == "__main__":
    main()

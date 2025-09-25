"""
Advanced Face Landmark Detection using MediaPipe Face Landmarker
Optimized for live video streaming with 478 3D landmarks and blendshapes.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
import json
import time
import threading
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)


class LiveFaceLandmarkerDetector:
    """
    Advanced face landmark detector using MediaPipe Face Landmarker task API.
    Optimized for live video streaming with async processing.
    """
    
    def __init__(self, 
                 model_path: str = None,
                 min_face_detection_confidence: float = 0.5,
                 min_face_presence_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5,
                 num_faces: int = 1,
                 output_blendshapes: bool = True,
                 result_callback: Optional[Callable] = None):
        """
        Initialize MediaPipe Face Landmarker for live streaming.
        
        Args:
            model_path: Path to face landmarker model file
            min_face_detection_confidence: Minimum confidence for face detection (0.0-1.0)
            min_face_presence_confidence: Minimum confidence for face presence (0.0-1.0)
            min_tracking_confidence: Minimum confidence for face tracking (0.0-1.0)
            num_faces: Maximum number of faces to detect
            output_blendshapes: Whether to output blendshape scores for expressions
            result_callback: Callback function for async results
        """
        self.mp_tasks = mp.tasks
        self.mp_vision = mp.tasks.vision
        
        # Download model if not provided
        if model_path is None:
            model_path = self._download_model()
        
        self.model_path = model_path
        self.result_callback = result_callback or self._default_callback
        
        # Initialize detector with live stream configuration
        base_options = self.mp_tasks.BaseOptions(model_asset_path=model_path)
        
        # Configure for LIVE_STREAM mode as per documentation
        options = self.mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=self.mp_vision.RunningMode.LIVE_STREAM,  # Key: LIVE_STREAM mode
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence, 
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=output_blendshapes,
            output_facial_transformation_matrixes=True,
            result_callback=self._result_callback_wrapper
        )
        
        # Create the detector
        self.detector = self.mp_vision.FaceLandmarker.create_from_options(options)
        
        # Thread-safe result storage
        self._latest_result = None
        self._result_lock = threading.Lock()
        self._frame_counter = 0
        
        # Define eye and mouth landmark indices for EAR/MAR calculation
        # CORRECTED for MediaPipe Face Landmarker 478-point model
        self.landmark_indices = {
            # Left eye landmarks - CORRECTED MediaPipe 478-point indices for proper EAR calculation
            'left_eye': [33, 160, 158, 133, 153, 144],  # outer, top1, top2, inner, bottom1, bottom2
            # Right eye landmarks - CORRECTED MediaPipe 478-point indices for proper EAR calculation  
            'right_eye': [362, 385, 387, 263, 373, 380],  # outer, top1, top2, inner, bottom1, bottom2
            # Mouth landmarks for MAR calculation - MediaPipe 478-point mouth contour
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            # Nose landmarks for head pose estimation
            'nose': [1, 2, 5, 4, 6, 19, 20],  # Key nose points for pose calculation
            # Face outline for head pose estimation - key boundary points
            'face_outline': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323]
        }
        
        logger.info(f"✅ LiveFaceLandmarkerDetector initialized with {num_faces} faces, LIVE_STREAM mode")
    
    def _download_model(self) -> str:
        """Download the Face Landmarker model if not present."""
        import urllib.request
        import os
        
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "face_landmarker.task"
        
        if not model_path.exists():
            logger.info("📥 Downloading MediaPipe Face Landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, str(model_path))
            logger.info(f"✅ Model downloaded to {model_path}")
        
        return str(model_path)
    
    def _result_callback_wrapper(self, result, output_image: mp.Image, timestamp_ms: int):
        """
        Internal callback wrapper for async results.
        Called automatically by MediaPipe in LIVE_STREAM mode.
        """
        with self._result_lock:
            self._latest_result = {
                'result': result,
                'image': output_image, 
                'timestamp': timestamp_ms,
                'frame_id': self._frame_counter
            }
        
        # Call user-defined callback if provided
        if self.result_callback:
            try:
                self.result_callback(result, output_image, timestamp_ms)
            except Exception as e:
                logger.error(f"Error in result callback: {e}")
    
    def _default_callback(self, result, output_image: mp.Image, timestamp_ms: int):
        """Default callback that logs detection results."""
        if result.face_landmarks:
            logger.debug(f"🎯 Detected {len(result.face_landmarks)} face(s) at {timestamp_ms}ms")
            if result.face_blendshapes:
                # Log some key blendshapes for drowsiness
                blendshapes = result.face_blendshapes[0]
                eye_blink_left = next((b.score for b in blendshapes if b.category_name == 'eyeBlinkLeft'), 0)
                eye_blink_right = next((b.score for b in blendshapes if b.category_name == 'eyeBlinkRight'), 0) 
                jaw_open = next((b.score for b in blendshapes if b.category_name == 'jawOpen'), 0)
                logger.debug(f"   👁️ Blink L/R: {eye_blink_left:.3f}/{eye_blink_right:.3f}, 👄 Jaw: {jaw_open:.3f}")
    
    def process_frame_async(self, image: np.ndarray) -> None:
        """
        Process frame asynchronously for live streaming.
        Results are available via callback or get_latest_result().
        
        Args:
            image: Input image (BGR format)
        """
        try:
            # Convert BGR to RGB  
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Generate timestamp in milliseconds
            timestamp_ms = int(time.time() * 1000)
            
            # Process asynchronously - results come via callback
            self.detector.detect_async(mp_image, timestamp_ms)
            self._frame_counter += 1
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    def get_latest_result(self) -> Optional[Dict]:
        """
        Get the latest detection result from async processing.
        Thread-safe method to retrieve results.
        
        Returns:
            Latest result dictionary or None
        """
        with self._result_lock:
            return self._latest_result.copy() if self._latest_result else None
    
    def extract_landmarks_from_result(self, result_data: Dict) -> Optional[Dict]:
        """
        Extract landmark data from MediaPipe result for drowsiness detection.
        
        Args:
            result_data: Result from get_latest_result()
            
        Returns:
            Dictionary with landmark data compatible with existing system
        """
        if not result_data or not result_data['result'].face_landmarks:
            return None
        
        result = result_data['result']
        face_landmarks = result.face_landmarks[0]  # First face
        
        # Convert normalized landmarks to pixel coordinates
        # Note: MediaPipe Face Landmarker returns normalized coordinates (0-1)
        image_width = 640  # Default, should be updated with actual image dimensions
        image_height = 480
        
        landmarks = []
        for landmark in face_landmarks:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height) 
            z = landmark.z  # Relative depth
            landmarks.append((x, y, z))
        
        # Extract specific landmark groups using 478-point model indices
        landmark_data = {
            'all_landmarks': landmarks,
            'left_eye': [landmarks[i] for i in self.landmark_indices['left_eye']],
            'right_eye': [landmarks[i] for i in self.landmark_indices['right_eye']], 
            'mouth': [landmarks[i] for i in self.landmark_indices['mouth']],
            'nose': [landmarks[i] for i in self.landmark_indices['nose']],
            'face_outline': [landmarks[i] for i in self.landmark_indices['face_outline']],
            'image_shape': (image_height, image_width),
            'timestamp': result_data['timestamp'],
            'frame_id': result_data['frame_id']
        }
        
        # Add blendshape scores if available
        if result.face_blendshapes:
            blendshapes = result.face_blendshapes[0]
            landmark_data['blendshapes'] = {
                bs.category_name: bs.score for bs in blendshapes
            }
            
            # Extract key drowsiness-related blendshapes
            landmark_data['drowsiness_blendshapes'] = {
                'eye_blink_left': landmark_data['blendshapes'].get('eyeBlinkLeft', 0),
                'eye_blink_right': landmark_data['blendshapes'].get('eyeBlinkRight', 0), 
                'eye_squint_left': landmark_data['blendshapes'].get('eyeSquintLeft', 0),
                'eye_squint_right': landmark_data['blendshapes'].get('eyeSquintRight', 0),
                'jaw_open': landmark_data['blendshapes'].get('jawOpen', 0),
                'mouth_close': landmark_data['blendshapes'].get('mouthClose', 0)
            }
        
        return landmark_data
    
    def detect_landmarks_sync(self, image: np.ndarray) -> Optional[Dict]:
        """
        Synchronous landmark detection for compatibility with existing code.
        Note: This creates a temporary detector in IMAGE mode.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Landmark data dictionary or None
        """
        try:
            # Create temporary detector for single image processing
            base_options = self.mp_tasks.BaseOptions(model_asset_path=self.model_path)
            options = self.mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=self.mp_vision.RunningMode.IMAGE,  # IMAGE mode for sync
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True
            )
            
            with self.mp_vision.FaceLandmarker.create_from_options(options) as detector:
                # Convert and process
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                result = detector.detect(mp_image)
                
                if not result.face_landmarks:
                    return None
                
                # Create result data structure
                result_data = {
                    'result': result,
                    'timestamp': int(time.time() * 1000),
                    'frame_id': self._frame_counter
                }
                
                # Extract landmarks using same method
                landmarks = self.extract_landmarks_from_result(result_data)
                
                # Update image dimensions
                if landmarks:
                    h, w = image.shape[:2]
                    landmarks['image_shape'] = (h, w)
                    
                    # Recalculate coordinates with correct dimensions
                    face_landmarks = result.face_landmarks[0]
                    corrected_landmarks = []
                    for landmark in face_landmarks:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        z = landmark.z
                        corrected_landmarks.append((x, y, z))
                    
                    landmarks['all_landmarks'] = corrected_landmarks
                    landmarks['left_eye'] = [corrected_landmarks[i] for i in self.landmark_indices['left_eye']]
                    landmarks['right_eye'] = [corrected_landmarks[i] for i in self.landmark_indices['right_eye']]
                    landmarks['mouth'] = [corrected_landmarks[i] for i in self.landmark_indices['mouth']]
                    landmarks['nose'] = [corrected_landmarks[i] for i in self.landmark_indices['nose']]
                    landmarks['face_outline'] = [corrected_landmarks[i] for i in self.landmark_indices['face_outline']]
                
                return landmarks
                
        except Exception as e:
            logger.error(f"Error in sync detection: {e}")
            return None
    
    def close(self):
        """Clean up detector resources."""
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()
            logger.info("🛑 LiveFaceLandmarkerDetector closed")


# Compatibility wrapper for existing code
class FaceLandmarkDetector:
    """
    Wrapper class to maintain compatibility with existing code while using
    the new LiveFaceLandmarkerDetector internally.
    """
    
    def __init__(self, min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5,
                 min_face_detection_confidence: float = None,
                 min_face_presence_confidence: float = None,
                 **kwargs):
        """Initialize with backward compatibility."""
        # Handle both old and new parameter names
        face_detection_conf = min_face_detection_confidence or min_detection_confidence
        face_presence_conf = min_face_presence_confidence or min_detection_confidence
        
        # Remove conflicting parameters from kwargs if they exist to avoid multiple values error
        output_blendshapes = kwargs.pop('output_blendshapes', True)
        num_faces = kwargs.pop('num_faces', 1)
        result_callback = kwargs.pop('result_callback', None)
        
        self.detector = LiveFaceLandmarkerDetector(
            min_face_detection_confidence=face_detection_conf,
            min_face_presence_confidence=face_presence_conf,
            min_tracking_confidence=min_tracking_confidence,
            num_faces=num_faces,
            output_blendshapes=output_blendshapes,
            result_callback=result_callback,
            **kwargs
        )
        logger.info("✅ FaceLandmarkDetector wrapper initialized with new Face Landmarker")
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """Detect landmarks - compatible with existing interface."""
        return self.detector.detect_landmarks_sync(image)
    
    def draw_landmarks(self, image: np.ndarray, landmark_data: Dict, 
                      draw_all: bool = False) -> np.ndarray:
        """Draw landmarks on image - maintains existing interface."""
        if landmark_data is None:
            return image
        
        image_copy = image.copy()
        
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
                    cv2.circle(image_copy, (int(landmark[0]), int(landmark[1])), 2, color, -1)
        
        return image_copy


if __name__ == "__main__":
    # Demo of live face landmarker
    def demo_callback(result, output_image, timestamp_ms):
        if result.face_landmarks:
            print(f"🎯 Detected face with {len(result.face_landmarks[0])} landmarks at {timestamp_ms}ms")
            if result.face_blendshapes:
                blendshapes = result.face_blendshapes[0]
                eye_blink_left = next((b.score for b in blendshapes if b.category_name == 'eyeBlinkLeft'), 0)
                print(f"   👁️ Left eye blink: {eye_blink_left:.3f}")
    
    detector = LiveFaceLandmarkerDetector(result_callback=demo_callback)
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame asynchronously
            detector.process_frame_async(frame)
            
            # Get latest results  
            result = detector.get_latest_result()
            if result:
                landmarks = detector.extract_landmarks_from_result(result)
                if landmarks:
                    print(f"📊 Extracted {len(landmarks['all_landmarks'])} landmarks")
            
            cv2.imshow('Live Face Landmarker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

"""
Step 3: Feature Calculation
Calculate EAR, MAR, blink frequency, and head nod features from facial landmarks.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import math


class DrowsinessFeatureExtractor:
    """Extract drowsiness-related features from facial landmarks."""
    
    def __init__(self, ear_threshold: float = 0.25, mar_threshold: float = 0.5):
        """
        Initialize feature extractor.
        
        Args:
            ear_threshold: Threshold for eye aspect ratio (below = closed eyes)
            mar_threshold: Threshold for mouth aspect ratio (above = yawning)
        """
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        
        # Blink detection parameters  
        self.blink_buffer = deque(maxlen=15)  # Store last 15 EAR values (increased for better detection)
        self.blink_count = 0
        self.last_blink_time = 0
        self.session_start_time = None  # Track session start for proper frequency calculation
        
        # Head nod detection parameters
        self.head_angle_buffer = deque(maxlen=5)  # Store last 5 head angles
        self.nod_count = 0
        
        # Debug logging
        self.frame_count = 0
        
    def calculate_ear(self, eye_landmarks: List[Tuple[float, float, float]]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for drowsiness detection.
        
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        where p1-p6 are eye landmark points
        
        Args:
            eye_landmarks: List of eye landmark coordinates
            
        Returns:
            Eye aspect ratio (lower = more closed eyes)
        """
        if len(eye_landmarks) < 6:
            return 0.0
        
        # Use key points for EAR calculation
        # Simplified version using 6 key points
        p1 = np.array(eye_landmarks[0][:2])  # Left corner
        p2 = np.array(eye_landmarks[1][:2])  # Top
        p3 = np.array(eye_landmarks[2][:2])  # Top
        p4 = np.array(eye_landmarks[3][:2])  # Right corner
        p5 = np.array(eye_landmarks[4][:2])  # Bottom
        p6 = np.array(eye_landmarks[5][:2])  # Bottom
        
        # Calculate distances
        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        
        # Debug logging for EAR calculation
        if self.frame_count % 10 == 0:  # Log every 10th frame to avoid spam
            print(f"🔍 EAR Debug - Frame {self.frame_count}:")
            print(f"   Points: p1={p1}, p2={p2}, p3={p3}, p4={p4}, p5={p5}, p6={p6}")
            print(f"   Distances: v1={vertical_1:.3f}, v2={vertical_2:.3f}, h={horizontal:.3f}")
            print(f"   Formula: ({vertical_1:.3f} + {vertical_2:.3f}) / (2 * {horizontal:.3f}) = {(vertical_1 + vertical_2) / (2.0 * horizontal):.3f}")
        
        # Avoid division by zero
        if horizontal == 0:
            return 0.0
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        
        # Check for abnormal EAR values
        if ear > 1.0:  # Normal EAR should be < 1.0
            if self.frame_count % 20 == 0:  # Periodic warning
                print(f"⚠️  HIGH EAR WARNING: {ear:.3f} (normal: 0.2-0.4)")
                print(f"   This suggests coordinate scaling issues or wrong landmark selection")
        
        return ear
    
    def calculate_mar(self, mouth_landmarks: List[Tuple[float, float, float]]) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR) for yawn detection.
        
        Args:
            mouth_landmarks: List of mouth landmark coordinates
            
        Returns:
            Mouth aspect ratio (higher = more open mouth)
        """
        if len(mouth_landmarks) < 6:
            return 0.0
        
        # Use key mouth points
        p1 = np.array(mouth_landmarks[0][:2])  # Left corner
        p2 = np.array(mouth_landmarks[1][:2])  # Top
        p3 = np.array(mouth_landmarks[2][:2])  # Top
        p4 = np.array(mouth_landmarks[3][:2])  # Right corner
        p5 = np.array(mouth_landmarks[4][:2])  # Bottom
        p6 = np.array(mouth_landmarks[5][:2])  # Bottom
        
        # Calculate distances
        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        
        if horizontal == 0:
            return 0.0
        
        mar = (vertical_1 + vertical_2) / (2.0 * horizontal) * 1.5
        return mar
    
    def calculate_head_pose(self, nose_landmarks: List[Tuple[float, float, float]], 
                           face_outline: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Calculate head pose angles (pitch, yaw, roll) for head nod detection.
        
        Args:
            nose_landmarks: Nose landmark coordinates
            face_outline: Face outline landmark coordinates
            
        Returns:
            Dictionary with pitch, yaw, roll angles
        """
        if len(nose_landmarks) < 3 or len(face_outline) < 4:
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        
        # Simplified head pose calculation using key points
        # This is a basic implementation - for production, use more sophisticated methods
        
        # Use nose tip and face outline points
        nose_tip = np.array(nose_landmarks[0][:2])
        chin = np.array(face_outline[8][:2])  # Chin point
        left_ear = np.array(face_outline[0][:2])  # Left ear
        right_ear = np.array(face_outline[16][:2])  # Right ear
        
        # Calculate basic angles
        # Pitch: vertical angle (nodding up/down)
        pitch = math.atan2(chin[1] - nose_tip[1], abs(chin[0] - nose_tip[0]))
        
        # Yaw: horizontal angle (turning left/right)
        yaw = math.atan2(nose_tip[0] - (left_ear[0] + right_ear[0])/2, 
                        abs(nose_tip[1] - (left_ear[1] + right_ear[1])/2))
        
        # Roll: rotation angle (tilting left/right)
        roll = math.atan2(right_ear[1] - left_ear[1], right_ear[0] - left_ear[0])
        
        return {
            'pitch': math.degrees(pitch),
            'yaw': math.degrees(yaw), 
            'roll': math.degrees(roll)
        }
    
    def detect_blink(self, ear: float, timestamp: float) -> bool:
        """
        Detect if a blink occurred based on EAR values with improved logic.
        
        Args:
            ear: Current eye aspect ratio
            timestamp: Current timestamp
            
        Returns:
            True if blink detected, False otherwise
        """
        self.frame_count += 1
        
        # Initialize session start time
        if self.session_start_time is None:
            self.session_start_time = timestamp
            
        self.blink_buffer.append(ear)
        
        # Need at least 3 frames for detection (adjusted for 2 FPS)
        if len(self.blink_buffer) < 3:
            return False
        
        # Simplified blink detection for low FPS: look for significant drop
        # Pattern: HIGH → LOW → HIGH (center frame is blink)
        recent_values = list(self.blink_buffer)[-3:]
        
        # Adaptive threshold based on recent EAR values
        avg_ear = sum(recent_values) / len(recent_values)
        adaptive_threshold = min(self.ear_threshold, avg_ear * 0.7)  # 70% of average
        
        # Blink detection: significant drop in middle frame (3-frame pattern for 2 FPS)
        if (recent_values[0] > adaptive_threshold and
            recent_values[1] < adaptive_threshold and  # Blink frame  
            recent_values[2] > adaptive_threshold):
            
            # Avoid counting same blink multiple times
            if timestamp - self.last_blink_time > 0.2:  # 200ms minimum between blinks
                self.blink_count += 1
                self.last_blink_time = timestamp
                
                # Debug logging
                print(f"🔍 BLINK DETECTED! Frame {self.frame_count}, EAR pattern: {[f'{v:.3f}' for v in recent_values]} (threshold: {adaptive_threshold:.3f}, 2 FPS optimized)")
                
                return True
        
        return False
    
    def detect_head_nod(self, head_angles: Dict[str, float], timestamp: float) -> bool:
        """
        Detect head nodding (sudden pitch changes).
        
        Args:
            head_angles: Dictionary with pitch, yaw, roll angles
            timestamp: Current timestamp
            
        Returns:
            True if head nod detected, False otherwise
        """
        pitch = head_angles['pitch']
        self.head_angle_buffer.append(pitch)
        
        # Need at least 3 frames to detect nod
        if len(self.head_angle_buffer) < 3:
            return False
        
        # Nod detection: sudden change in pitch angle
        pitch_change = abs(self.head_angle_buffer[-1] - self.head_angle_buffer[-3])
        
        if pitch_change > 15:  # Threshold for significant head movement
            self.nod_count += 1
            return True
        
        return False
    
    def extract_features(self, landmark_data: Dict, timestamp: float = 0.0) -> Dict[str, float]:
        """
        Extract all drowsiness features from landmark data.
        
        Args:
            landmark_data: Landmark data from face detection
            timestamp: Current timestamp
            
        Returns:
            Dictionary with all extracted features
        """
        if landmark_data is None:
            return self._get_default_features()
        
        # Calculate basic ratios
        left_ear = self.calculate_ear(landmark_data.get('left_eye', []))
        right_ear = self.calculate_ear(landmark_data.get('right_eye', []))
        avg_ear = (left_ear + right_ear) / 2.0
        
        mar = self.calculate_mar(landmark_data.get('mouth', []))
        
        # Calculate head pose
        head_angles = self.calculate_head_pose(
            landmark_data.get('nose', []),
            landmark_data.get('face_outline', [])
        )
        
        # Detect events
        blink_detected = self.detect_blink(avg_ear, timestamp)
        nod_detected = self.detect_head_nod(head_angles, timestamp)
        
        # Calculate blink frequency (blinks per minute) based on session duration
        if self.session_start_time and timestamp > self.session_start_time:
            session_duration_minutes = (timestamp - self.session_start_time) / 60.0
            blink_frequency = self.blink_count / max(session_duration_minutes, 0.1)  # Minimum 6 seconds
        else:
            blink_frequency = 0.0
        
        # Calculate nod frequency (nods per minute)
        nod_frequency = self.nod_count / max(timestamp / 60.0, 1.0) if timestamp > 0 else 0
        
        # Calculate eye closure percentage (adaptive to actual EAR range)
        # For high EAR values, use relative closure based on recent averages
        if len(self.blink_buffer) >= 3:
            recent_ears = list(self.blink_buffer)[-10:]  # Last 10 EAR values
            max_ear = max(recent_ears)  # Eyes most open
            min_ear = min(recent_ears)  # Eyes most closed
            ear_range = max_ear - min_ear
            
            if ear_range > 0.1:  # If there's significant variation
                # Normalize current EAR relative to observed range
                normalized_openness = (avg_ear - min_ear) / ear_range
                eye_closure = 1.0 - max(0.0, min(1.0, normalized_openness))
            else:
                # Fallback: use traditional method with adaptive threshold
                adaptive_threshold = max(self.ear_threshold, avg_ear * 0.7)
                eye_closure = max(0.0, 1.0 - (avg_ear / adaptive_threshold))
        else:
            # Not enough data yet
            eye_closure = 0.0
        
        # Debug logging for eye closure
        print(f"👁️  EAR Debug: avg={avg_ear:.3f}, closure={eye_closure:.3f}, buffer_size={len(self.blink_buffer)}")
        
        # Calculate yawn indicator
        yawn_indicator = 1.0 if mar > self.mar_threshold else 0.0
        
        features = {
            # Basic ratios
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear,
            'mar': mar,
            
            # Head pose
            'head_pitch': head_angles['pitch'],
            'head_yaw': head_angles['yaw'],
            'head_roll': head_angles['roll'],
            
            # Event detection
            'blink_detected': 1.0 if blink_detected else 0.0,
            'nod_detected': 1.0 if nod_detected else 0.0,
            
            # Frequency metrics
            'blink_frequency': blink_frequency,
            'nod_frequency': nod_frequency,
            
            # Derived features
            'eye_closure': eye_closure,
            'yawn_indicator': yawn_indicator,
            
            # Temporal features
            'timestamp': timestamp,
            
            # Face detection confidence
            'face_detected': 1.0 if landmark_data else 0.0
        }
        
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when no face is detected."""
        return {
            'left_ear': 0.0,
            'right_ear': 0.0,
            'avg_ear': 0.0,
            'mar': 0.0,
            'head_pitch': 0.0,
            'head_yaw': 0.0,
            'head_roll': 0.0,
            'blink_detected': 0.0,
            'nod_detected': 0.0,
            'blink_frequency': 0.0,
            'nod_frequency': 0.0,
            'eye_closure': 1.0,  # Eyes closed if no face detected
            'yawn_indicator': 0.0,
            'timestamp': 0.0,
            'face_detected': 0.0
        }
    
    def reset_counters(self):
        """Reset blink and nod counters."""
        self.blink_count = 0
        self.nod_count = 0
        self.last_blink_time = 0
        self.blink_buffer.clear()
        self.head_angle_buffer.clear()
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return [
            'left_ear', 'right_ear', 'avg_ear', 'mar',
            'head_pitch', 'head_yaw', 'head_roll',
            'blink_detected', 'nod_detected',
            'blink_frequency', 'nod_frequency',
            'eye_closure', 'yawn_indicator',
            'face_detected'
        ]


def main():
    """Demo the feature extraction."""
    extractor = DrowsinessFeatureExtractor()
    
    # Create sample landmark data
    sample_landmarks = {
        'left_eye': [(100, 100, 0), (110, 95, 0), (120, 95, 0), (130, 100, 0), (120, 105, 0), (110, 105, 0)],
        'right_eye': [(200, 100, 0), (210, 95, 0), (220, 95, 0), (230, 100, 0), (220, 105, 0), (210, 105, 0)],
        'mouth': [(150, 200, 0), (160, 195, 0), (170, 195, 0), (180, 200, 0), (170, 205, 0), (160, 205, 0)],
        'nose': [(165, 150, 0), (170, 160, 0), (175, 150, 0)],
        'face_outline': [(100, 80, 0), (200, 80, 0), (220, 120, 0), (100, 120, 0)]
    }
    
    # Extract features
    features = extractor.extract_features(sample_landmarks, timestamp=1.0)
    
    print("Extracted features:")
    for name, value in features.items():
        print(f"  {name}: {value:.3f}")
    
    print(f"\nFeature names: {extractor.get_feature_names()}")
    print("Feature extraction module ready!")


if __name__ == "__main__":
    main()

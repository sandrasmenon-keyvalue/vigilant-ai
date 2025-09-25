"""
Enhanced Feature Extraction using MediaPipe Face Landmarker Blendshapes
Produces the same 14 features but with improved accuracy using facial expressions.
NO MODEL RETRAINING REQUIRED - same feature interface, better values!
"""

import math
import time
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np

class EnhancedDrowsinessFeatureExtractor:
    """
    Enhanced feature extractor that uses MediaPipe Face Landmarker blendshapes
    to improve the accuracy of the same 14 features expected by the trained model.
    """
    
    def __init__(self, ear_threshold: float = 0.25, mar_threshold: float = 0.5):
        """
        Initialize enhanced feature extractor.
        
        Args:
            ear_threshold: Traditional EAR threshold (will be adaptively adjusted)
            mar_threshold: Traditional MAR threshold (will be adaptively adjusted)
        """
        # Traditional parameters (kept for compatibility)
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        
        # Enhanced parameters for MediaPipe Face Landmarker
        self.blendshape_blink_threshold = 0.4  # eyeBlinkLeft/Right threshold
        self.blendshape_yawn_threshold = 0.3   # jawOpen threshold
        self.blendshape_squint_threshold = 0.2  # eyeSquintLeft/Right threshold
        
        # Temporal tracking (same as before)
        self.blink_count = 0
        self.nod_count = 0
        self.last_blink_time = 0
        self.frame_count = 0
        self.session_start_time = None
        
        # Buffers for temporal analysis (optimized for MediaPipe)
        self.blink_buffer = deque(maxlen=3)  # For 2 FPS
        self.head_angle_buffer = deque(maxlen=5)
        self.blendshape_buffer = deque(maxlen=5)  # NEW: Track blendshape history
        
        print("✅ Enhanced Feature Extractor initialized with MediaPipe blendshape integration")
    
    def calculate_ear(self, eye_landmarks: List[Tuple[int, int, int]]) -> float:
        """Calculate Eye Aspect Ratio - same logic as before."""
        if len(eye_landmarks) < 6:
            return 0.0
        
        # Use first 6 landmarks for EAR calculation
        p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]
        
        # Vertical distances
        vertical_1 = math.sqrt((p2[0] - p6[0])**2 + (p2[1] - p6[1])**2)
        vertical_2 = math.sqrt((p3[0] - p5[0])**2 + (p3[1] - p5[1])**2)
        
        # Horizontal distance  
        horizontal = math.sqrt((p1[0] - p4[0])**2 + (p1[1] - p4[1])**2)
        
        if horizontal == 0:
            return 0.0
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def calculate_mar(self, mouth_landmarks: List[Tuple[int, int, int]]) -> float:
        """Calculate Mouth Aspect Ratio - same logic as before."""
        if len(mouth_landmarks) < 6:
            return 0.0
        
        # Use key mouth points
        points = mouth_landmarks[:6]  
        
        # Vertical distances (mouth height)
        vertical_1 = math.sqrt((points[1][0] - points[5][0])**2 + (points[1][1] - points[5][1])**2)
        vertical_2 = math.sqrt((points[2][0] - points[4][0])**2 + (points[2][1] - points[4][1])**2)
        
        # Horizontal distance (mouth width)
        horizontal = math.sqrt((points[0][0] - points[3][0])**2 + (points[0][1] - points[3][1])**2)
        
        if horizontal == 0:
            return 0.0
        
        mar = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return mar
    
    def calculate_head_pose(self, nose_landmarks: List, face_outline: List) -> Dict[str, float]:
        """Calculate head pose angles - same logic as before."""
        if len(nose_landmarks) < 2 or len(face_outline) < 3:
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        
        nose_tip = nose_landmarks[0]
        nose_bridge = nose_landmarks[1]  
        left_ear = face_outline[0]
        right_ear = face_outline[1]
        
        # Pitch: vertical angle (nodding up/down)
        pitch = math.atan2(nose_bridge[1] - nose_tip[1], 
                          abs(nose_bridge[0] - nose_tip[0]))
        
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
    
    def enhanced_blink_detection(self, ear: float, blendshapes: Dict[str, float], timestamp: float) -> bool:
        """
        Enhanced blink detection using BOTH EAR and MediaPipe blendshapes.
        Returns the SAME blink_detected feature but with better accuracy.
        """
        self.frame_count += 1
        
        if self.session_start_time is None:
            self.session_start_time = timestamp
        
        self.blink_buffer.append(ear)
        
        # Method 1: Traditional EAR-based detection (improved for MediaPipe)
        ear_blink = False
        if len(self.blink_buffer) >= 3:
            recent_values = list(self.blink_buffer)[-3:]
            
            # Adaptive threshold for CORRECTED MediaPipe EAR values
            avg_ear = sum(recent_values) / len(recent_values)
            # Adaptive threshold for normal EAR range (CORRECTED)
            adaptive_threshold = min(max(avg_ear * 0.8, self.ear_threshold), avg_ear * 0.9)  # Reasonable range
            
            # Traditional pattern detection
            if (recent_values[0] > adaptive_threshold and
                recent_values[1] < adaptive_threshold and
                recent_values[2] > adaptive_threshold):
                ear_blink = True
        
        # Method 2: MediaPipe blendshape-based detection (NEW!)
        blendshape_blink = False
        if blendshapes:
            left_blink = blendshapes.get('eyeBlinkLeft', 0.0)
            right_blink = blendshapes.get('eyeBlinkRight', 0.0)
            
            # Detect blink if either eye shows strong blink signal
            max_blink_score = max(left_blink, right_blink)
            if max_blink_score > self.blendshape_blink_threshold:
                blendshape_blink = True
                
                print(f"🎭 BLENDSHAPE BLINK! L:{left_blink:.3f}, R:{right_blink:.3f} (frame {self.frame_count})")
        
        # Method 3: Hybrid approach - combine both methods
        blink_detected = ear_blink or blendshape_blink
        
        # Update counters (same logic as before)
        if blink_detected and timestamp - self.last_blink_time > 0.2:
            self.blink_count += 1
            self.last_blink_time = timestamp
            
            method = "EAR+Blendshape" if ear_blink and blendshape_blink else ("EAR" if ear_blink else "Blendshape")
            print(f"✅ ENHANCED BLINK DETECTED! Method: {method}, Frame: {self.frame_count}")
            
            return True
        
        return False
    
    def enhanced_yawn_detection(self, mar: float, blendshapes: Dict[str, float]) -> bool:
        """
        Enhanced yawn detection using BOTH MAR and MediaPipe blendshapes.
        Returns the SAME yawn_indicator feature but with better accuracy.
        """
        # Method 1: Traditional MAR-based detection
        mar_yawn = mar > self.mar_threshold
        
        # Method 2: MediaPipe blendshape-based detection (NEW!)
        blendshape_yawn = False
        if blendshapes:
            jaw_open = blendshapes.get('jawOpen', 0.0)
            if jaw_open > self.blendshape_yawn_threshold:
                blendshape_yawn = True
                print(f"🎭 BLENDSHAPE YAWN! jawOpen: {jaw_open:.3f}")
        
        # Method 3: Hybrid approach
        yawn_detected = mar_yawn or blendshape_yawn
        
        if yawn_detected:
            method = "MAR+Blendshape" if mar_yawn and blendshape_yawn else ("MAR" if mar_yawn else "Blendshape")
            print(f"🥱 ENHANCED YAWN DETECTED! Method: {method}, MAR: {mar:.3f}")
        
        return yawn_detected
    
    def enhanced_eye_closure(self, avg_ear: float, blendshapes: Dict[str, float]) -> float:
        """
        Enhanced eye closure calculation using blendshapes for validation.
        Returns the SAME eye_closure feature but with better accuracy.
        """
        # Method 1: Traditional EAR-based closure (adaptive for MediaPipe)
        if len(self.blink_buffer) >= 3:
            recent_ears = list(self.blink_buffer)[-10:]
            max_ear = max(recent_ears) if recent_ears else avg_ear
            min_ear = min(recent_ears) if recent_ears else avg_ear
            ear_range = max_ear - min_ear
            
            if ear_range > 0.1:
                normalized_openness = (avg_ear - min_ear) / ear_range
                ear_closure = 1.0 - max(0.0, min(1.0, normalized_openness))
            else:
                # Adaptive threshold for CORRECTED MediaPipe EAR values
                adaptive_threshold = max(avg_ear * 0.85, self.ear_threshold * 1.2)  # Reasonable baseline
                ear_closure = max(0.0, 1.0 - (avg_ear / adaptive_threshold))
        else:
            ear_closure = 0.0
        
        # Method 2: Blendshape validation (NEW!)
        if blendshapes:
            left_blink = blendshapes.get('eyeBlinkLeft', 0.0)
            right_blink = blendshapes.get('eyeBlinkRight', 0.0)
            left_squint = blendshapes.get('eyeSquintLeft', 0.0)
            right_squint = blendshapes.get('eyeSquintRight', 0.0)
            
            # Average closure from blendshapes
            blendshape_closure = (left_blink + right_blink + left_squint + right_squint) / 4.0
            
            # Use blendshape as validation/enhancement
            if blendshape_closure > 0.3:  # Strong blendshape signal
                # Blend EAR and blendshape methods
                enhanced_closure = (ear_closure * 0.6) + (blendshape_closure * 0.4)
                print(f"👁️  ENHANCED CLOSURE: EAR={ear_closure:.3f}, Blendshape={blendshape_closure:.3f} → {enhanced_closure:.3f}")
                return enhanced_closure
        
        return ear_closure
    
    def detect_head_nod(self, head_angles: Dict[str, float], timestamp: float) -> bool:
        """Head nod detection - same logic as before."""
        pitch = head_angles['pitch']
        self.head_angle_buffer.append(pitch)
        
        if len(self.head_angle_buffer) < 3:
            return False
        
        pitch_change = abs(self.head_angle_buffer[-1] - self.head_angle_buffer[-3])
        
        if pitch_change > 15:  # Threshold for significant head movement
            self.nod_count += 1
            return True
        
        return False
    
    def extract_features(self, landmark_data: Dict, timestamp: float = 0.0) -> Dict[str, float]:
        """
        Extract the SAME 14 features but with enhanced accuracy using MediaPipe blendshapes.
        NO MODEL RETRAINING REQUIRED - same interface, better values!
        """
        if landmark_data is None:
            return self._get_default_features()
        
        # Extract blendshapes if available (NEW!)
        blendshapes = landmark_data.get('blendshapes', {})
        drowsiness_blendshapes = landmark_data.get('drowsiness_blendshapes', {})
        
        # Use available blendshapes
        available_blendshapes = blendshapes or drowsiness_blendshapes
        
        # Calculate basic ratios (same as before)
        left_ear = self.calculate_ear(landmark_data.get('left_eye', []))
        right_ear = self.calculate_ear(landmark_data.get('right_eye', []))
        avg_ear = (left_ear + right_ear) / 2.0 if left_ear > 0 and right_ear > 0 else 0.0
        mar = self.calculate_mar(landmark_data.get('mouth', []))
        
        # Calculate head pose (same as before)
        head_angles = self.calculate_head_pose(
            landmark_data.get('nose', []),
            landmark_data.get('face_outline', [])
        )
        
        # ENHANCED DETECTION using blendshapes!
        blink_detected = self.enhanced_blink_detection(avg_ear, available_blendshapes, timestamp)
        nod_detected = self.detect_head_nod(head_angles, timestamp)
        yawn_indicator = 1.0 if self.enhanced_yawn_detection(mar, available_blendshapes) else 0.0
        eye_closure = self.enhanced_eye_closure(avg_ear, available_blendshapes)
        
        # Calculate frequencies (same logic as before)
        if self.session_start_time and timestamp > self.session_start_time:
            session_duration_minutes = (timestamp - self.session_start_time) / 60.0
            blink_frequency = self.blink_count / max(session_duration_minutes, 0.1)
        else:
            blink_frequency = 0.0
        
        nod_frequency = self.nod_count / max(timestamp / 60.0, 1.0) if timestamp > 0 else 0.0
        
        # Return the SAME 14 features (no model retraining needed!)
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
            
            # Event detection (ENHANCED!)
            'blink_detected': 1.0 if blink_detected else 0.0,
            'nod_detected': 1.0 if nod_detected else 0.0,
            
            # Frequency metrics (ENHANCED!)
            'blink_frequency': blink_frequency,
            'nod_frequency': nod_frequency,
            
            # Derived features (ENHANCED!)
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
        self.blendshape_buffer.clear()
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names - SAME 14 features as before!"""
        return [
            'left_ear', 'right_ear', 'avg_ear', 'mar',
            'head_pitch', 'head_yaw', 'head_roll', 
            'blink_detected', 'nod_detected',
            'blink_frequency', 'nod_frequency',
            'eye_closure', 'yawn_indicator',
            'face_detected'
        ]

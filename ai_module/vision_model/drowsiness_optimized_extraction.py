"""
Drowsiness-Optimized Feature Extraction
Specifically tuned to generate stronger drowsiness signals for better model predictions.
"""

from enhanced_feature_extraction import EnhancedDrowsinessFeatureExtractor
import math
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np

class DrowsinessOptimizedExtractor(EnhancedDrowsinessFeatureExtractor):
    """
    Optimized version that generates stronger drowsiness signals by:
    1. More sensitive thresholds
    2. Better temporal detection
    3. Enhanced combination of features
    4. Focused on what the XGBoost model needs
    """
    
    def __init__(self, ear_threshold: float = 0.25, mar_threshold: float = 0.5):
        super().__init__(ear_threshold, mar_threshold)
        
        # More sensitive thresholds for drowsiness detection
        self.blendshape_blink_threshold = 0.3   # Lower = more sensitive  
        self.blendshape_yawn_threshold = 0.25   # Lower = more sensitive
        self.blendshape_squint_threshold = 0.15 # Detect fatigue squinting
        
        # Drowsiness-specific parameters (adjusted for CORRECT EAR values ~0.2-0.4)
        self.prolonged_closure_threshold = 0.8  # Seconds for prolonged closure
        self.slow_blink_threshold = 8.0         # Blinks/min = drowsy
        self.microsleep_ear_threshold = 0.15    # EAR for microsleep detection (CORRECTED)
        
        # Enhanced temporal tracking
        self.ear_history = deque(maxlen=10)     # Track EAR trends  
        self.closure_duration = 0.0             # Track how long eyes closed
        self.last_closure_start = 0.0
        self.prolonged_closures = 0
        
        print("🎯 Drowsiness-Optimized Extractor: More sensitive detection enabled")
    
    def detect_prolonged_eye_closure(self, avg_ear: float, timestamp: float) -> float:
        """
        Detect prolonged eye closures (key drowsiness indicator).
        Returns enhanced eye_closure value that better indicates fatigue.
        """
        self.ear_history.append(avg_ear)
        
        # Define closure based on CORRECTED MediaPipe EAR values
        # For corrected MediaPipe: EAR ~0.25+ = open, EAR ~0.2- = closing, EAR ~0.15- = closed
        closure_threshold = 0.2  # CORRECTED for proper MediaPipe landmark indices
        
        is_closing = avg_ear < closure_threshold
        
        if is_closing:
            if self.last_closure_start == 0.0:
                self.last_closure_start = timestamp
            self.closure_duration = timestamp - self.last_closure_start
        else:
            if self.closure_duration > self.prolonged_closure_threshold:
                self.prolonged_closures += 1
                print(f"💤 PROLONGED CLOSURE DETECTED! Duration: {self.closure_duration:.2f}s")
            self.last_closure_start = 0.0
            self.closure_duration = 0.0
        
        # Enhanced eye closure calculation (CORRECTED for normal EAR values)
        if len(self.ear_history) >= 5:
            recent_ears = list(self.ear_history)[-5:]
            avg_recent = sum(recent_ears) / len(recent_ears)
            
            # More aggressive closure detection for drowsiness (CORRECTED thresholds)
            if avg_recent < 0.15:  # Very closed (eyes nearly shut)
                enhanced_closure = 0.9
            elif avg_recent < 0.2:   # Moderately closed (drowsy)
                enhanced_closure = 0.7  
            elif avg_recent < 0.25:  # Slightly closed (fatigue)
                enhanced_closure = 0.4
            else:
                enhanced_closure = max(0.0, (0.4 - avg_recent) / 0.2)  # Gradual from normal range
            
            # Boost closure value if prolonged
            if self.closure_duration > 0.5:  # Eyes closed for >0.5s
                enhanced_closure = min(1.0, enhanced_closure + 0.3)
                print(f"👁️ FATIGUE CLOSURE: EAR={avg_ear:.3f}, enhanced={enhanced_closure:.3f}, duration={self.closure_duration:.1f}s")
            
            return enhanced_closure
        
        return super().enhanced_eye_closure(avg_ear, {})
    
    def detect_microsleep_patterns(self, avg_ear: float, timestamp: float) -> bool:
        """
        Detect microsleep patterns - brief periods where eyes close involuntarily.
        This is a strong drowsiness indicator.
        """
        if len(self.ear_history) < 5:
            return False
        
        recent_ears = list(self.ear_history)[-5:]
        
        # Pattern: normal -> sudden drop -> recovery (microsleep) - CORRECTED thresholds
        if (recent_ears[0] > 0.23 and  # Was normal (CORRECTED)
            recent_ears[1] > 0.23 and  # Still normal (CORRECTED)
            recent_ears[2] < 0.18 and  # Sudden drop (microsleep) (CORRECTED)
            recent_ears[3] < 0.18 and  # Still low (CORRECTED)
            recent_ears[4] > 0.22):    # Recovery (CORRECTED)
            
            print(f"💤 MICROSLEEP PATTERN DETECTED! EAR sequence: {[f'{e:.2f}' for e in recent_ears]}")
            return True
        
        return False
    
    def enhanced_yawn_detection_aggressive(self, mar: float, blendshapes: Dict[str, float]) -> bool:
        """
        More aggressive yawn detection - catch subtle yawns.
        """
        # Method 1: Lower MAR threshold  
        aggressive_mar_threshold = 0.3  # Lower than default 0.5
        mar_yawn = mar > aggressive_mar_threshold
        
        # Method 2: MediaPipe blendshapes (more sensitive)
        blendshape_yawn = False
        if blendshapes:
            jaw_open = blendshapes.get('jawOpen', 0.0)
            mouth_close = blendshapes.get('mouthClose', 0.0)
            
            # Detect yawn with lower threshold OR mouth opening pattern
            if jaw_open > 0.2 or (jaw_open > 0.15 and mouth_close < 0.3):
                blendshape_yawn = True
                print(f"🥱 AGGRESSIVE YAWN! jawOpen: {jaw_open:.3f}, mouthClose: {mouth_close:.3f}")
        
        return mar_yawn or blendshape_yawn
    
    def calculate_fatigue_indicators(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate additional fatigue indicators that boost drowsiness scores.
        """
        # Heavy eyelids indicator (combination of EAR and blinking patterns)
        avg_ear = features['avg_ear']
        blink_freq = features['blink_frequency']
        
        heavy_eyelids = 0.0
        if avg_ear < 0.25 and blink_freq > 25:  # Low EAR + frequent blinking = fatigue (CORRECTED)
            heavy_eyelids = min(1.0, (30 - blink_freq) / 20 + (0.3 - avg_ear) * 5)  # CORRECTED calculation
            if heavy_eyelids > 0.5:
                print(f"😴 HEAVY EYELIDS DETECTED! EAR={avg_ear:.3f}, freq={blink_freq:.1f}")
        
        # Slow response indicator (based on head movement)
        head_movement = abs(features['head_pitch']) + abs(features['head_yaw']) + abs(features['head_roll'])
        slow_response = 1.0 if head_movement < 2.0 else 0.0  # Very little head movement
        
        return {
            'heavy_eyelids': heavy_eyelids,
            'slow_response': slow_response
        }
    
    def extract_features(self, landmark_data: Dict, timestamp: float = 0.0) -> Dict[str, float]:
        """
        Extract optimized features with stronger drowsiness signals.
        """
        if landmark_data is None:
            return self._get_default_features()
        
        # Get base features from enhanced extractor
        features = super().extract_features(landmark_data, timestamp)
        
        # Apply drowsiness optimizations
        avg_ear = features['avg_ear']
        blendshapes = landmark_data.get('blendshapes', {})
        
        # 1. Enhanced eye closure with prolonged detection
        optimized_eye_closure = self.detect_prolonged_eye_closure(avg_ear, timestamp)
        features['eye_closure'] = optimized_eye_closure
        
        # 2. Detect microsleep patterns
        microsleep = self.detect_microsleep_patterns(avg_ear, timestamp)
        if microsleep:
            # Boost relevant features when microsleep detected
            features['eye_closure'] = min(1.0, features['eye_closure'] + 0.4)
            features['blink_detected'] = 1.0
        
        # 3. More aggressive yawn detection
        mar = features['mar']
        aggressive_yawn = self.enhanced_yawn_detection_aggressive(mar, blendshapes)
        features['yawn_indicator'] = 1.0 if aggressive_yawn else 0.0
        
        # 4. Adjust blink frequency interpretation for drowsiness
        blink_freq = features['blink_frequency']
        if blink_freq > 30:  # Very frequent blinking = fatigue
            print(f"⚡ HIGH BLINK FREQUENCY: {blink_freq:.1f}/min (fatigue indicator)")
        elif blink_freq < 8:  # Very slow blinking = drowsiness
            print(f"💤 LOW BLINK FREQUENCY: {blink_freq:.1f}/min (drowsiness indicator)")
            # Boost eye closure for slow blinking
            features['eye_closure'] = min(1.0, features['eye_closure'] + 0.2)
        
        # 5. Head position drowsiness indicators
        head_pitch = features['head_pitch']
        if head_pitch < -10:  # Head dropping significantly
            features['nod_detected'] = 1.0
            print(f"🎪 HEAD DROP DETECTED: pitch={head_pitch:.1f}°")
        
        # 6. Calculate fatigue indicators
        fatigue_indicators = self.calculate_fatigue_indicators(features)
        
        # 7. Apply fatigue boost to key features
        if fatigue_indicators['heavy_eyelids'] > 0.5:
            features['eye_closure'] = min(1.0, features['eye_closure'] + 0.3)
            print(f"😴 FATIGUE BOOST APPLIED: eye_closure boosted to {features['eye_closure']:.3f}")
        
        return features
    
    def reset_counters(self):
        """Reset all counters including new fatigue tracking."""
        super().reset_counters()
        self.ear_history.clear()
        self.closure_duration = 0.0
        self.last_closure_start = 0.0
        self.prolonged_closures = 0

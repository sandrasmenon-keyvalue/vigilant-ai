"""
Synchronized Inference Engine
Receives DV (vision data) and HV (vitals data) with timestamps, synchronizes them,
and passes synchronized data to health score calculation and other checks.
"""

import json
import time
import threading
import queue
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from collections import deque
import logging

# Import health score calculation
from actions.health_score_prediction.health_score_calculation import calculate_health_score

# Import alert prediction
from actions.alert_prediction.alert_prediction import create_alert, Alert

# Import simple WebSocket sender for sending health scores and alerts
from websocket.simple_websocket_sender import send_health_score_simple, send_alert_simple
from websocket.websocket_client import VitalsWebSocketClient

logger = logging.getLogger(__name__)


@dataclass
class DataPoint:
    """Container for incoming data points with timestamp."""
    timestamp: float
    data: Any
    data_type: str  # 'dv' or 'hv'
    source: str
    received_at: float


@dataclass
class SynchronizedResult:
    """Result of synchronized inference."""
    timestamp: float
    dv_score: float
    hv_score: float
    health_score: float
    dv_data: Any
    hv_data: Any
    sync_tolerance: float
    processing_time: float
    mode: str = "synchronized"  # "synchronized", "dv_only", "hv_only"


class SynchronizedInferenceEngine:
    """
    Inference engine that synchronizes DV and HV data by timestamp
    and processes them together for health score calculation.
    """
    
    def __init__(self, 
                 sync_tolerance: float = 0.1,
                 max_buffer_size: int = 1000,
                 sync_timeout: float = 2.0,
                 health_score_callback: Optional[Callable] = None,
                 alert_callback: Optional[Callable] = None,
                 enable_logging: bool = True):
        """
        Initialize the synchronized inference engine.
        
        Args:
            sync_tolerance: Maximum time difference for synchronization (seconds)
            max_buffer_size: Maximum number of data points to buffer
            sync_timeout: Maximum time to wait for missing data type (seconds)
            health_score_callback: Optional callback function for health scores
            alert_callback: Optional callback function for alerts
            enable_logging: Enable/disable logging
        """
        self.sync_tolerance = sync_tolerance
        self.max_buffer_size = max_buffer_size
        self.sync_timeout = sync_timeout
        self.health_score_callback = health_score_callback
        self.alert_callback = alert_callback
        self.enable_logging = enable_logging
        
        # Data buffers for DV and HV data
        self.dv_buffer = deque(maxlen=max_buffer_size)
        self.hv_buffer = deque(maxlen=max_buffer_size)
        
        # Timeout tracking for single data type processing
        self.pending_dv_timeout = {}  # timestamp -> timeout_time
        self.pending_hv_timeout = {}  # timestamp -> timeout_time
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Rule check parameters
        self.dv_threshold = 0.6  # DV alert threshold
        self.hv_threshold = 0.3  # HV alert threshold (adjust as needed)
        self.hr_normal_range = (60, 100)  # Normal HR range (BPM)
        self.spo2_normal_range = (95, 100)  # Normal SpO2 range (%)
        self.monitoring_window = 30.0  # 30 seconds monitoring window
        self.max_violations = 21  # Maximum violations in 30s window
        
        # Rule check counters and timers
        self.hr_violation_counter = 0
        self.spo2_violation_counter = 0
        self.hr_window_start = time.time()
        self.spo2_window_start = time.time()
        
        # Processing statistics
        self.stats = {
            'total_dv_received': 0,
            'total_hv_received': 0,
            'total_synchronized': 0,
            'total_health_scores': 0,
            'total_alerts': 0,
            'dv_alerts': 0,
            'hv_alerts': 0,
            'hr_alerts': 0,
            'spo2_alerts': 0,
            'buffer_overflows': 0,
            'sync_failures': 0,
            'processing_errors': 0
        }
        
        # Results storage
        self.results_history = deque(maxlen=100)
        
        logger.info(f"✅ Synchronized Inference Engine initialized")
        logger.info(f"   Sync tolerance: {sync_tolerance}s")
        logger.info(f"   Buffer size: {max_buffer_size}")
        logger.info(f"   DV threshold: {self.dv_threshold}")
        logger.info(f"   HV threshold: {self.hv_threshold}")
        logger.info(f"   HR normal range: {self.hr_normal_range}")
        logger.info(f"   SpO2 normal range: {self.spo2_normal_range}")
        logger.info(f"   Monitoring window: {self.monitoring_window}s")
        logger.info(f"   Max violations: {self.max_violations}")
        logger.info(f"   Logging: {'Enabled' if enable_logging else 'Disabled'}")
    
    def receive_dv_data(self, dv_data: Any, timestamp: float, source: str = "vision") -> bool:
        """
        Receive DV (vision) data with timestamp.
        
        DV DATA (drowsiness_score, timestamp) ARRIVES HERE FOR SYNCHRONOUS PROCESSING
        This is where DV data with timestamp gets buffered and synchronized with HV data.
        
        Args:
            dv_data: Vision data (image path, numpy array, or processed features)
            timestamp: Timestamp when data was collected
            source: Source of the data
            
        Returns:
            True if data was received successfully, False otherwise
        """
        try:
            print("***************** DV DATA RECEIVED *****************", dv_data)
            with self.lock:
                # TODO: DV data (dv,tk) gets stored in buffer for synchronization
                data_point = DataPoint(
                    timestamp=timestamp,
                    data=dv_data,
                    data_type='dv',
                    source=source,
                    received_at=time.time()
                )
                
                self.dv_buffer.append(data_point)
                self.stats['total_dv_received'] += 1
                
                # Set timeout for this DV data point
                self.pending_dv_timeout[timestamp] = time.time() + self.sync_timeout
                
                if self.enable_logging:
                    logger.info(f"📸 Received DV data at timestamp {timestamp:.3f}, waiting {self.sync_timeout}s for HV data")
                    logger.info(f"   DV Buffer size: {len(self.dv_buffer)}, HV Buffer size: {len(self.hv_buffer)}")
                
                # Try to synchronize with HV data
                self._try_synchronize()
                
                return True
                
        except Exception as e:
            self.stats['processing_errors'] += 1
            logger.error(f"Error receiving DV data: {e}")
            return False
    
    def receive_dv_from_inference_api(self, drowsiness_score: float, timestamp: float, 
                                    features: Dict[str, float] = None, 
                                    alert_level: str = None, 
                                    confidence: float = None,
                                    frame_id: str = None) -> bool:
        """
        Receive DV data from inference_api.py process_frame_inference result.
        
        Args:
            drowsiness_score: Drowsiness score from vision processing (0-1)
            timestamp: Frame timestamp
            features: Extracted features from vision processing
            alert_level: Alert level from vision processing
            confidence: Prediction confidence
            frame_id: Frame identifier
            
        Returns:
            True if data was received successfully, False otherwise
        """
        # Create DV data structure compatible with existing system
        dv_data = {
            'drowsiness_score': drowsiness_score,
            'features': features or {},
            'alert_level': alert_level or 'low',
            'confidence': confidence or 0.0,
            'frame_id': frame_id,
            'source': 'inference_api'
        }
        
        # Use existing receive_dv_data method
        success = self.receive_dv_data(dv_data, timestamp, source="inference_api")
        
        if success and self.enable_logging:
            logger.info(f"📹 DV data from inference_api: drowsiness_score={drowsiness_score:.3f}, "
                       f"alert_level={alert_level}, confidence={confidence:.3f}")
        
        return success
    
    def receive_hv_data(self, hv_data: Any, timestamp: float, source: str = "vitals") -> bool:
        """
        Receive HV (vitals) data with timestamp.
        
        Args:
            hv_data: Vitals data (dictionary with HR, SpO2, etc.)
            timestamp: Timestamp when data was collected
            source: Source of the data
            
        Returns:
            True if data was received successfully, False otherwise
        """
        try:
            with self.lock:
                data_point = DataPoint(
                    timestamp=timestamp,
                    data=hv_data,
                    data_type='hv',
                    source=source,
                    received_at=time.time()
                )
                
                self.hv_buffer.append(data_point)
                self.stats['total_hv_received'] += 1
                
                # Set timeout for this HV data point
                self.pending_hv_timeout[timestamp] = time.time() + self.sync_timeout
                
                if self.enable_logging:
                    logger.info(f"📊 Received HV data at timestamp {timestamp:.3f}, waiting {self.sync_timeout}s for DV data")
                    logger.info(f"   DV Buffer size: {len(self.dv_buffer)}, HV Buffer size: {len(self.hv_buffer)}")
                
                # Try to synchronize with DV data
                self._try_synchronize()
                
                return True
                
        except Exception as e:
            self.stats['processing_errors'] += 1
            logger.error(f"Error receiving HV data: {e}")
            return False
    
    def _try_synchronize(self):
        """Try to synchronize DV and HV data by timestamp with timeout-based fallback."""
        current_time = time.time()
        
        # First, check for timed-out data points and process them individually
        self._process_timed_out_data(current_time)
        
        # Then try to find synchronized pairs
        synchronized_pairs = []
        
        # Create copies to avoid modifying buffers during iteration
        dv_list = list(self.dv_buffer)
        hv_list = list(self.hv_buffer)
        
        # Find matching pairs within sync tolerance
        for dv_point in dv_list:
            for hv_point in hv_list:
                time_diff = abs(dv_point.timestamp - hv_point.timestamp)
                
                if time_diff <= self.sync_tolerance:
                    synchronized_pairs.append((dv_point, hv_point, time_diff))
        
        # Process synchronized pairs
        for dv_point, hv_point, time_diff in synchronized_pairs:
            try:
                result = self._process_synchronized_data(dv_point, hv_point, time_diff)
                if result:
                    self.results_history.append(result)
                    self.stats['total_synchronized'] += 1
                    
                    # Call health score callback if provided
                    if self.health_score_callback:
                        try:
                            self.health_score_callback(result)
                        except Exception as e:
                            logger.error(f"Error in health score callback: {e}")
                
            except Exception as e:
                self.stats['processing_errors'] += 1
                logger.error(f"Error processing synchronized data: {e}")
        
        # Remove processed data points from buffers
        self._cleanup_processed_data(synchronized_pairs)
    
    def _process_timed_out_data(self, current_time: float):
        """Process data points that have timed out waiting for synchronization."""
        # Check for timed-out DV data points
        timed_out_dv = []
        for timestamp, timeout_time in list(self.pending_dv_timeout.items()):
            if current_time >= timeout_time:
                timed_out_dv.append(timestamp)
                del self.pending_dv_timeout[timestamp]
        
        # Process timed-out DV data with default HV
        for timestamp in timed_out_dv:
            # Find the DV data point
            dv_point = None
            for point in self.dv_buffer:
                if abs(point.timestamp - timestamp) < 0.001:  # Find matching timestamp
                    dv_point = point
                    break
            
            if dv_point:
                # Create default HV data point
                default_hv_point = DataPoint(
                    timestamp=dv_point.timestamp,
                    data=0.0,  # Default HV score when no vitals
                    data_type='hv',
                    source='default',
                    received_at=time.time()
                )
                
                if self.enable_logging:
                    logger.info(f"⏰ DV data timed out, processing with default HV: DV={dv_point.data}")
                
                # Process the timed-out data
                self._process_single_data_type(dv_point, default_hv_point, "dv_only")
        
        # Check for timed-out HV data points
        timed_out_hv = []
        for timestamp, timeout_time in list(self.pending_hv_timeout.items()):
            if current_time >= timeout_time:
                timed_out_hv.append(timestamp)
                del self.pending_hv_timeout[timestamp]
        
        # Process timed-out HV data with default DV
        for timestamp in timed_out_hv:
            # Find the HV data point
            hv_point = None
            for point in self.hv_buffer:
                if abs(point.timestamp - timestamp) < 0.001:  # Find matching timestamp
                    hv_point = point
                    break
            
            if hv_point:
                # Create default DV data point
                default_dv_point = DataPoint(
                    timestamp=hv_point.timestamp,
                    data=0.0,  # Default DV score when no video
                    data_type='dv',
                    source='default',
                    received_at=time.time()
                )
                
                if self.enable_logging:
                    logger.info(f"⏰ HV data timed out, processing with default DV: HV={hv_point.data}")
                
                # Process the timed-out data
                self._process_single_data_type(default_dv_point, hv_point, "hv_only")
    
    def _process_single_data_type(self, dv_point: DataPoint, hv_point: DataPoint, mode: str):
        """Process data when only one type is available after timeout."""
        try:
            result = self._process_synchronized_data(dv_point, hv_point, 0.0)
            if result:
                self.results_history.append(result)
                self.stats['total_synchronized'] += 1
                
                # Add mode information to result
                result.mode = mode
                
                # Call health score callback if provided
                if self.health_score_callback:
                    try:
                        self.health_score_callback(result)
                    except Exception as e:
                        logger.error(f"Error in health score callback: {e}")
            
        except Exception as e:
            self.stats['processing_errors'] += 1
            logger.error(f"Error processing single data type ({mode}): {e}")
    
    def _process_synchronized_data(self, dv_point: DataPoint, hv_point: DataPoint, time_diff: float) -> Optional[SynchronizedResult]:
        """
        Process synchronized DV and HV data.
        
        Args:
            dv_point: DV data point
            hv_point: HV data point
            time_diff: Time difference between the two points
            
        Returns:
            SynchronizedResult or None if processing fails
        """
        start_time = time.time()
        
        try:
            # Extract DV score from vision data
            dv_score = self._extract_dv_score(dv_point.data)
            
            # Extract HV score from vitals data
            hv_score = self._extract_hv_score(hv_point.data)
            
            # Calculate health score using the existing function
            health_score = calculate_health_score(dv_score, hv_score)
            
            # Send health score via WebSocket
            try:
                socket = VitalsWebSocketClient()
                message = json.dumps({
                    "payload": {
                        "score": health_score,
                        "reasons": []
                    }
                })
                socket.send_message(message)
                # success = send_health_score_simple(
                #     health_score=int(health_score),
                #     dv_score=dv_score,
                #     hv_score=hv_score,
                #     mode="synchronized",
                #     sync_tolerance=time_diff,
                #     source="synchronized_inference_engine"
                # )
                # if success and self.enable_logging:
                #     logger.info(f"📤 Health score queued for WebSocket: {health_score:.3f}")
                # elif not success:
                #     logger.warning(f"⚠️  Failed to queue health score: {health_score:.3f}")
            except Exception as e:
                logger.error(f"❌ Error queuing health score for WebSocket: {e}")
            
            processing_time = time.time() - start_time
            
            result = SynchronizedResult(
                timestamp=(dv_point.timestamp + hv_point.timestamp) / 2,  # Average timestamp
                dv_score=dv_score,
                hv_score=hv_score,
                health_score=health_score,
                dv_data=dv_point.data,
                hv_data=hv_point.data,
                sync_tolerance=time_diff,
                processing_time=processing_time,
                mode="synchronized"
            )
            
            self.stats['total_health_scores'] += 1
            
            # Perform rule checks after health score calculation
            self._perform_rule_checks(result, dv_point, hv_point)
            
            if self.enable_logging:
                logger.info(f"🔄 Synchronized: DV={dv_score:.3f}, HV={hv_score:.3f}, Health={health_score:.3f} (Δt={time_diff:.3f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing synchronized data: {e}")
            return None
    
    def _extract_dv_score(self, dv_data: Any) -> float:
        """
        Extract DV score from vision data.
        
        Args:
            dv_data: Vision data (could be image path, numpy array, or pre-computed score)
            
        Returns:
            DV score (0-1)
        """
        # If dv_data is already a score (float)
        if isinstance(dv_data, (int, float)):
            return float(dv_data)
        
        # If dv_data is a dictionary with score
        if isinstance(dv_data, dict):
            # Handle drowsiness_score from inference_api
            if 'drowsiness_score' in dv_data:
                return float(dv_data['drowsiness_score'])
            elif 'dv_score' in dv_data:
                return float(dv_data['dv_score'])
            elif 'score' in dv_data:
                return float(dv_data['score'])
        
        # If dv_data is an image path or array, you would process it here
        # For now, return a default value
        logger.warning(f"Could not extract DV score from data type: {type(dv_data)}")
        return 0.5  # Default middle value
    
    def _extract_hv_score(self, hv_data: Any) -> float:
        """
        Extract HV score from vitals data.
        
        Args:
            hv_data: Vitals data (could be dictionary with HR, SpO2, or pre-computed score)
            
        Returns:
            HV score (0-1)
        """
        # If hv_data is already a score (float)
        if isinstance(hv_data, (int, float)):
            return float(hv_data)
        
        # If hv_data is a dictionary with score
        if isinstance(hv_data, dict):
            if 'hv_score' in hv_data:
                return float(hv_data['hv_score'])
            elif 'score' in hv_data:
                return float(hv_data['score'])
            elif 'health_score' in hv_data:
                return float(hv_data['health_score'])
        
        # If hv_data contains raw vitals, you would process it here
        # For now, return a default value
        logger.warning(f"Could not extract HV score from data type: {type(hv_data)}")
        return 0.5  # Default middle value
    
    def _perform_rule_checks(self, result: SynchronizedResult, dv_point: DataPoint, hv_point: DataPoint):
        """
        Perform rule checks on the synchronized result.
        
        Args:
            result: Synchronized result with DV, HV, and health score
            dv_point: DV data point
            hv_point: HV data point
        """
        try:
            # Extract vitals data for rule checks
            hr, spo2 = self._extract_vitals_data(hv_point.data)
            
            # Rule 1: Check DV threshold
            if result.dv_score > self.dv_threshold:
                self._trigger_alert("dv_high", "Driver appears drowsy. Please take a break.")
                self.stats['dv_alerts'] += 1
            
            # Rule 2: Check HV threshold
            if result.hv_score < self.hv_threshold:
                self._trigger_alert("hv_low", "Health indicators are concerning. Please rest.")
                self.stats['hv_alerts'] += 1
            
            # Rule 3: Check HR violations with 30s counter
            if hr is not None:
                self._check_hr_violations(hr, result.timestamp)
            
            # Rule 4: Check SpO2 violations with 30s counter
            if spo2 is not None:
                self._check_spo2_violations(spo2, result.timestamp)
            
            # Rule 5: Check environmental conditions
            self._check_environmental_conditions(result)
                
        except Exception as e:
            logger.error(f"Error in rule checks: {e}")
    
    def _extract_vitals_data(self, hv_data: Any) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract HR and SpO2 from HV data.
        
        Args:
            hv_data: HV data (could be dict, float, or other format)
            
        Returns:
            Tuple of (hr, spo2) or (None, None) if not found
        """
        hr = None
        spo2 = None
        
        if isinstance(hv_data, dict):
            # Try to extract HR and SpO2 from dictionary
            hr = hv_data.get('hr', hv_data.get('hr_median', None))
            spo2 = hv_data.get('spo2', hv_data.get('spo2_median', None))
            
            # Convert to float if possible
            try:
                hr = float(hr) if hr is not None else None
                spo2 = float(spo2) if spo2 is not None else None
            except (ValueError, TypeError):
                hr = None
                spo2 = None
        
        return hr, spo2
    
    def _check_environmental_conditions(self, result: SynchronizedResult):
        """
        Check environmental conditions and trigger alerts if necessary.
        
        Args:
            result: SynchronizedResult containing environmental data
        """
        try:
            # Check if environmental conditions are available
            if not hasattr(result, 'environmental_conditions') or result.environmental_conditions is None:
                return
            
            env_conditions = result.environmental_conditions
            
            # Check for critical environmental conditions
            if env_conditions.get('overall_risk') == 'critical':
                self._trigger_alert("environmental_critical", 
                                  "Critical environmental conditions detected. Immediate attention required.")
                self.stats['environmental_alerts'] = self.stats.get('environmental_alerts', 0) + 1
            
            # Check for environmental alerts
            alerts = env_conditions.get('alerts', [])
            for alert in alerts:
                if 'temperature' in alert.lower():
                    if 'critical' in alert.lower():
                        self._trigger_alert("temperature_critical", alert)
                    else:
                        self._trigger_alert("temperature_warning", alert)
                elif 'co2' in alert.lower():
                    if 'dangerous' in alert.lower():
                        self._trigger_alert("co2_dangerous", alert)
                    elif 'concerning' in alert.lower():
                        self._trigger_alert("co2_concerning", alert)
                    else:
                        self._trigger_alert("co2_elevated", alert)
                
                self.stats['environmental_alerts'] = self.stats.get('environmental_alerts', 0) + 1
                
        except Exception as e:
            logger.error(f"Error checking environmental conditions: {e}")
    
    def _check_hr_violations(self, hr: float, timestamp: float):
        """
        Check HR violations with 30-second monitoring window.
        
        Args:
            hr: Heart rate value
            timestamp: Current timestamp
        """
        current_time = time.time()
        
        # Check if 30 seconds have passed since window start
        if current_time - self.hr_window_start >= self.monitoring_window:
            # Reset counter and start new window
            self.hr_violation_counter = 0
            self.hr_window_start = current_time
            if self.enable_logging:
                logger.info("🔄 HR monitoring window reset")
        
        # Check if HR is outside normal range
        if hr < self.hr_normal_range[0] or hr > self.hr_normal_range[1]:
            self.hr_violation_counter += 1
            
            if self.enable_logging:
                logger.warning(f"⚠️  HR violation #{self.hr_violation_counter}: {hr} BPM (normal: {self.hr_normal_range[0]}-{self.hr_normal_range[1]})")
            
            # Check if violations exceed threshold
            if self.hr_violation_counter > self.max_violations:
                self._trigger_alert("hr_violations", "Heart rate is abnormal. Please check your condition.")
                self.stats['hr_alerts'] += 1
                # Reset counter after alert
                self.hr_violation_counter = 0
                self.hr_window_start = current_time
    
    def _check_spo2_violations(self, spo2: float, timestamp: float):
        """
        Check SpO2 violations with 30-second monitoring window.
        
        Args:
            spo2: SpO2 value
            timestamp: Current timestamp
        """
        current_time = time.time()
        
        # Check if 30 seconds have passed since window start
        if current_time - self.spo2_window_start >= self.monitoring_window:
            # Reset counter and start new window
            self.spo2_violation_counter = 0
            self.spo2_window_start = current_time
            if self.enable_logging:
                logger.info("🔄 SpO2 monitoring window reset")
        
        # Check if SpO2 is outside normal range
        if spo2 < self.spo2_normal_range[0] or spo2 > self.spo2_normal_range[1]:
            self.spo2_violation_counter += 1
            
            if self.enable_logging:
                logger.warning(f"⚠️  SpO2 violation #{self.spo2_violation_counter}: {spo2}% (normal: {self.spo2_normal_range[0]}-{self.spo2_normal_range[1]})")
            
            # Check if violations exceed threshold
            if self.spo2_violation_counter > self.max_violations:
                self._trigger_alert("spo2_violations", "Oxygen level is low. Please check the ambient or breathe deeply and rest.")
                self.stats['spo2_alerts'] += 1
                # Reset counter after alert
                self.spo2_violation_counter = 0
                self.spo2_window_start = current_time
    
    def _trigger_alert(self, alert_type: str, reason: str):
        """
        Trigger alert using alert_prediction module and send via WebSocket.
        
        Args:
            alert_type: Type of alert (dv_high, hv_low, hr_violations, spo2_violations)
            reason: Detailed reason for the alert
        """
        self.stats['total_alerts'] += 1
        
        # Create alert using alert_prediction module
        alert = create_alert(reason)
        
        # Determine alert level based on alert type
        alert_level = "warning"
        if "critical" in alert_type.lower() or "violations" in alert_type:
            alert_level = "critical"
        elif "environmental" in alert_type and "critical" in alert_type:
            alert_level = "critical"
        elif "dangerous" in alert_type:
            alert_level = "critical"
        
        # Send alert via WebSocket
        try:
            # success = send_alert_simple(
            #     alert_type=alert_type,
            #     reason=reason,
            #     alert_level=alert_level,
            #     source="synchronized_inference_engine",
            #     timestamp=time.time()
            # )
            # if success and self.enable_logging:
            #     logger.info(f"📤 Alert queued for WebSocket: {alert_type}")
            # elif not success:
            #     logger.warning(f"⚠️  Failed to queue alert: {alert_type}")

            socket = VitalsWebSocketClient()
            message = json.dumps({
                "type": "speech_alert",
                "userId": "user123",
                "payload": {
                    "text": reason,
                    "description":""
                }
            })
            socket.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error queuing alert for WebSocket: {e}")
        
        if self.enable_logging:
            logger.warning(f"🚨 ALERT [{alert_type}]: {alert.reason}")
            logger.warning(f"   Alert Type: {alert.type}")
            logger.warning(f"   Alert Level: {alert_level}")
        
        # Call alert callback if provided (pass the Alert object)
        if self.alert_callback:
            try:
                self.alert_callback(alert, alert_type)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def get_rule_check_status(self) -> Dict[str, Any]:
        """Get current status of rule checks."""
        current_time = time.time()
        
        return {
            'dv_threshold': self.dv_threshold,
            'hv_threshold': self.hv_threshold,
            'hr_normal_range': self.hr_normal_range,
            'spo2_normal_range': self.spo2_normal_range,
            'monitoring_window': self.monitoring_window,
            'max_violations': self.max_violations,
            'hr_violation_counter': self.hr_violation_counter,
            'spo2_violation_counter': self.spo2_violation_counter,
            'hr_window_remaining': max(0, self.monitoring_window - (current_time - self.hr_window_start)),
            'spo2_window_remaining': max(0, self.monitoring_window - (current_time - self.spo2_window_start)),
            'alert_stats': {
                'total_alerts': self.stats['total_alerts'],
                'dv_alerts': self.stats['dv_alerts'],
                'hv_alerts': self.stats['hv_alerts'],
                'hr_alerts': self.stats['hr_alerts'],
                'spo2_alerts': self.stats['spo2_alerts']
            }
        }
    
    def _cleanup_processed_data(self, synchronized_pairs: List[Tuple]):
        """Remove processed data points from buffers."""
        processed_dv_timestamps = {pair[0].timestamp for pair in synchronized_pairs}
        processed_hv_timestamps = {pair[1].timestamp for pair in synchronized_pairs}
        
        # Remove processed DV points
        self.dv_buffer = deque(
            [point for point in self.dv_buffer if point.timestamp not in processed_dv_timestamps],
            maxlen=self.max_buffer_size
        )
        
        # Remove processed HV points
        self.hv_buffer = deque(
            [point for point in self.hv_buffer if point.timestamp not in processed_hv_timestamps],
            maxlen=self.max_buffer_size
        )
    
    def get_latest_result(self) -> Optional[SynchronizedResult]:
        """Get the latest synchronized result."""
        if self.results_history:
            return self.results_history[-1]
        return None
    
    def get_results_history(self, count: int = 10) -> List[SynchronizedResult]:
        """Get recent synchronized results."""
        return list(self.results_history)[-count:] if self.results_history else []
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status and statistics."""
        with self.lock:
            return {
                'dv_buffer_size': len(self.dv_buffer),
                'hv_buffer_size': len(self.hv_buffer),
                'results_count': len(self.results_history),
                'stats': self.stats.copy()
            }
    
    def clear_buffers(self):
        """Clear all data buffers."""
        with self.lock:
            self.dv_buffer.clear()
            self.hv_buffer.clear()
            self.results_history.clear()
        
        logger.info("🧹 All buffers cleared")
    
    def get_health_score_trends(self, window_size: int = 10) -> Dict[str, Any]:
        """Analyze health score trends over recent window."""
        if len(self.results_history) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        recent_results = list(self.results_history)[-window_size:]
        health_scores = [r.health_score for r in recent_results]
        timestamps = [r.timestamp for r in recent_results]
        
        # Calculate trend
        if len(health_scores) >= 2:
            trend = "increasing" if health_scores[-1] > health_scores[0] else "decreasing"
            avg_score = sum(health_scores) / len(health_scores)
            min_score = min(health_scores)
            max_score = max(health_scores)
        else:
            trend = "stable"
            avg_score = health_scores[0] if health_scores else 0
            min_score = avg_score
            max_score = avg_score
        
        return {
            'trend': trend,
            'average_score': avg_score,
            'min_score': min_score,
            'max_score': max_score,
            'score_range': max_score - min_score,
            'window_size': len(health_scores),
            'timestamps': timestamps,
            'health_scores': health_scores
        }


def health_score_callback(result: SynchronizedResult) -> None:
    """
    Example callback function for health scores.
    
    Args:
        result: Synchronized inference result
    """
    print(f"🔔 Health Score: {result.health_score:.3f} at {result.timestamp:.3f}")
    print(f"   DV: {result.dv_score:.3f}, HV: {result.hv_score:.3f}")
    print(f"   Sync tolerance: {result.sync_tolerance:.3f}s")
    
    # Your business logic here
    if result.health_score > 0.8:
        print("   ✅ Excellent Health")
    elif result.health_score > 0.6:
        print("   ✅ Good Health")
    elif result.health_score > 0.4:
        print("   ⚠️  Fair Health")
    else:
        print("   🚨 Poor Health - Alert!")


def alert_callback(alert: Alert, alert_type: str) -> None:
    """
    Example callback function for alerts.
    
    Args:
        alert: Alert object from alert_prediction module
        alert_type: Type of alert (dv_high, hv_low, etc.)
    """
    print(f"🚨 Alert Callback: {alert_type}")
    print(f"   Type: {alert.type}")
    print(f"   Reason: {alert.reason}")


def main():
    """Demo the synchronized inference engine."""
    print("Synchronized Inference Engine Demo")
    print("=" * 50)
    
    # Initialize the engine
    engine = SynchronizedInferenceEngine(
        sync_tolerance=0.1,
        max_buffer_size=100,
        health_score_callback=health_score_callback,
        alert_callback=alert_callback,
        enable_logging=True
    )
    
    # Simulate receiving data
    print("\n🔄 Simulating data reception...")
    
    # Data points with timestamps (including some that will trigger alerts)
    data_points = [
        # Timestamp 1: Both DV and HV (normal)
        {'timestamp': 1703123456.000, 'dv': 0.3, 'hv': 0.7},
        
        # Timestamp 2: High DV score (should trigger DV alert)
        {'timestamp': 1703123457.000, 'dv': 0.8, 'hv': None},
        
        # Timestamp 3: Low HV score (should trigger HV alert)
        {'timestamp': 1703123458.000, 'dv': None, 'hv': 0.2},
        
        # Timestamp 4: Both DV and HV (slightly different times)
        {'timestamp': 1703123459.000, 'dv': 0.2, 'hv': 0.8},
        {'timestamp': 1703123459.050, 'dv': None, 'hv': 0.8},  # 50ms later
        
        # Timestamp 5: Both high DV and low HV (should trigger both alerts)
        {'timestamp': 1703123460.000, 'dv': 0.7, 'hv': 0.1},
    ]
    
    # Send data
    for i, point in enumerate(data_points):
        print(f"\n📊 Sending data point {i+1}:")
        print(f"   Timestamp: {point['timestamp']}")
        
        # Send DV data if available
        if point['dv'] is not None:
            success = engine.receive_dv_data(point['dv'], point['timestamp'], "vision")
            print(f"   DV Data: {'✅ Received' if success else '❌ Failed'}")
        
        # Send HV data if available
        if point['hv'] is not None:
            success = engine.receive_hv_data(point['hv'], point['timestamp'], "vitals")
            print(f"   HV Data: {'✅ Received' if success else '❌ Failed'}")
        
        # Small delay
        time.sleep(0.1)
    
    # Show results
    print(f"\n📈 Results:")
    results = engine.get_results_history()
    for i, result in enumerate(results, 1):
        print(f"   {i}. Health Score: {result.health_score:.3f} (DV: {result.dv_score:.3f}, HV: {result.hv_score:.3f})")
    
    # Show trends
    trends = engine.get_health_score_trends()
    print(f"\n📊 Health Score Trends:")
    print(f"   Trend: {trends['trend']}")
    print(f"   Average: {trends['average_score']:.3f}")
    print(f"   Range: {trends['min_score']:.3f} - {trends['max_score']:.3f}")
    
    # Show status
    status = engine.get_buffer_status()
    print(f"\n📋 Engine Status:")
    print(f"   DV Buffer: {status['dv_buffer_size']}")
    print(f"   HV Buffer: {status['hv_buffer_size']}")
    print(f"   Results: {status['results_count']}")
    print(f"   Total Synchronized: {status['stats']['total_synchronized']}")
    
    print("\n✅ Synchronized inference engine demo completed!")


if __name__ == "__main__":
    main()

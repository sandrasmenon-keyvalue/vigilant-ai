"""
Live Stream FastAPI Service for Vigilant AI Drowsiness Detection
Handles real-time camera streams with continuous drowsiness monitoring.
"""

import io
import os
import sys
import time
import asyncio
import base64
import logging
import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any, AsyncGenerator
from datetime import datetime
from collections import deque
import traceback
import csv
import os
import threading

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import requests
import aiohttp
import mediapipe as mp
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

# Import synchronized inference engine
try:
    from inference.synchronized_inference_engine import SynchronizedInferenceEngine
except ImportError as e:
    logging.error(f"Failed to import synchronized inference engine: {e}")
    SynchronizedInferenceEngine = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediaPipeFaceDetector:
    """MediaPipe face detection for live stream preprocessing."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect_face_quality(self, image_base64: str) -> dict:
        """
        Check face detection quality before sending to inference.
        
        Returns:
            dict: {
                'face_detected': bool,
                'face_confidence': float,
                'face_bbox': tuple,
                'landmark_count': int
            }
        """
        try:
            # Decode base64 image
            if image_base64.startswith('data:image'):
                image_base64 = image_base64.split(',')[1]
            
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmark_count = len(face_landmarks.landmark)
                
                # Calculate bounding box
                h, w = image_np.shape[:2]
                x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]
                
                bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                
                # Calculate face confidence based on landmark distribution
                face_confidence = min(1.0, landmark_count / 468.0)  # MediaPipe has 468 landmarks
                
                return {
                    'face_detected': True,
                    'face_confidence': face_confidence,
                    'face_bbox': bbox,
                    'landmark_count': landmark_count,
                    'face_size': (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
                }
            else:
                return {
                    'face_detected': False,
                    'face_confidence': 0.0,
                    'face_bbox': None,
                    'landmark_count': 0,
                    'face_size': 0
                }
                
        except Exception as e:
            logger.error(f"MediaPipe face detection error: {e}")
            return {
                'face_detected': False,
                'face_confidence': 0.0,
                'face_bbox': None,
                'landmark_count': 0,
                'face_size': 0,
                'error': str(e)
            }


# Initialize MediaPipe face detector
face_detector = MediaPipeFaceDetector()

# Create FastAPI app
app = FastAPI(
    title="Vigilant AI - Live Stream Drowsiness Detection",
    description="Real-time drowsiness detection from live camera streams",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Live stream management
active_streams: Dict[str, Dict] = {}
camera_manager = None
inference_client = None
sync_inference_engine = None

# Configuration
FRAME_RATE = 5  # Process every 5th frame for efficiency
WINDOW_SIZE_SECONDS = 5.0
MAX_STREAM_DURATION = 3600  # 1 hour max stream
CLEANUP_INTERVAL = 300  # Cleanup every 5 minutes
INFERENCE_API_URL = "http://localhost:8002"  # Inference API endpoint

# Global service start time
service_start_time = time.time()


class StreamConfig(BaseModel):
    """Configuration for live stream."""
    stream_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    camera_source: int = Field(default=0, description="Camera index (0 for default webcam)")
    frame_width: int = Field(default=640, description="Frame width")
    frame_height: int = Field(default=480, description="Frame height")
    processing_fps: int = Field(default=5, description="Processing frame rate")
    enable_display: bool = Field(default=False, description="Enable visual display")


class DrowsinessAlert(BaseModel):
    """Real-time drowsiness alert."""
    stream_id: str
    timestamp: str
    drowsiness_score: float
    confidence: float
    is_drowsy: bool
    alert_level: str  # "low", "medium", "high", "critical", "error", "no_face", "processing"
    features: Dict[str, Any]  # Allow mixed types for features and error info
    frame_count: int


class StreamStatus(BaseModel):
    """Stream status information."""
    stream_id: str
    status: str  # "active", "paused", "stopped", "error"
    start_time: str
    frames_processed: int
    alerts_triggered: int
    current_drowsiness_score: float
    avg_processing_time: float


class InferenceAPIClient:
    """Client for communicating with the Inference API."""
    
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip('/')
        self.session = None
    
    async def initialize(self):
        """Initialize the HTTP session."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        # Test connection to inference API
        try:
            async with self.session.get(f"{self.api_url}/inference/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    logger.info(f"✅ Connected to Inference API: {health_data.get('status')}")
                    return True
                else:
                    logger.error(f"❌ Inference API health check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to Inference API: {e}")
            return False
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
    
    async def infer_single_frame(self, image_base64: str, timestamp: float, frame_id: str) -> Dict[str, Any]:
        """Send single frame to inference API."""
        if not self.session:
            await self.initialize()
        
        try:
            frame_data = {
                "image_base64": image_base64,
                "timestamp": timestamp,
                "frame_id": frame_id
            }
            
            async with self.session.post(
                f"{self.api_url}/inference/single_frame",
                json=frame_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Inference API error {response.status}: {error_text}")
                    return {"error": f"API error {response.status}"}
        
        except asyncio.TimeoutError:
            logger.error("Inference API request timeout")
            return {"error": "Request timeout"}
        except Exception as e:
            logger.error(f"Inference API request failed: {e}")
            return {"error": str(e)}
    
    async def batch_infer_frames(self, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send batch of frames to inference API."""
        if not self.session:
            await self.initialize()
        
        try:
            # Convert frames to the correct format expected by inference API
            frame_data_list = []
            for frame in frames:
                frame_data_list.append({
                    "image_base64": frame["image_base64"],
                    "timestamp": frame["timestamp"],
                    "frame_id": frame.get("frame_id", f"frame_{len(frame_data_list)}")
                })
            
            video_data = {
                "frames": frame_data_list,
                "processing_options": {
                    "source": "live_stream",
                    "batch_size": len(frame_data_list)
                }
            }
            
            logger.info(f"📦 Sending batch of {len(frame_data_list)} frames to inference API")
            logger.info(f"   🎬 Frame IDs: {[f['frame_id'] for f in frame_data_list]}")
            logger.info(f"   ⏰ Time range: {min(f['timestamp'] for f in frame_data_list):.3f} - {max(f['timestamp'] for f in frame_data_list):.3f}")
            
            async with self.session.post(
                f"{self.api_url}/inference/video_stream",
                json=video_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result_data = await response.json()
                    logger.info(f"✅ Batch inference successful:")
                    logger.info(f"   📊 Avg drowsiness score: {result_data.get('avg_drowsiness_score', 0):.4f}")
                    logger.info(f"   📈 Max drowsiness score: {result_data.get('max_drowsiness_score', 0):.4f}")
                    logger.info(f"   🚨 Alert frames: {result_data.get('alert_frames', 0)}/{result_data.get('total_frames', 0)}")
                    logger.info(f"   ⏱️  Processing time: {result_data.get('processing_time', 0):.3f}s")
                    logger.info(f"   🎯 Processing FPS: {result_data.get('summary', {}).get('processing_fps', 0):.1f}")
                    return {
                        "success": True,
                        "data": result_data
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Batch inference error {response.status}: {error_text}")
                    return {
                        "success": False,
                        "error": f"API error {response.status}: {error_text}"
                    }
        
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class CameraManager:
    """Manages camera input and frame processing."""
    
    def __init__(self):
        self.cameras = {}
        self.frame_processors = {}
    
    def initialize_camera(self, stream_id: str, camera_source: int, width: int, height: int):
        """Initialize camera for a stream."""
        try:
            cap = cv2.VideoCapture(camera_source)
            if not cap.isOpened():
                raise Exception(f"Cannot open camera {camera_source}")
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)  # High capture rate, but we'll process at lower rate
            
            self.cameras[stream_id] = cap
            
            # Initialize frame processor for this stream
            self.frame_processors[stream_id] = StreamFrameProcessor(stream_id)
            
            logger.info(f"✅ Camera initialized for stream {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize camera: {e}")
            return False
    
    def get_frame(self, stream_id: str) -> Optional[np.ndarray]:
        """Get frame from camera."""
        if stream_id not in self.cameras:
            return None
        
        cap = self.cameras[stream_id]
        ret, frame = cap.read()
        
        if ret:
            return frame
        return None
    
    def release_camera(self, stream_id: str):
        """Release camera resources."""
        if stream_id in self.cameras:
            self.cameras[stream_id].release()
            del self.cameras[stream_id]
            logger.info(f"📹 Camera released for stream {stream_id}")
        
        if stream_id in self.frame_processors:
            del self.frame_processors[stream_id]


class StreamFrameProcessor:
    """Processes frames for a specific stream using MediaPipe + Inference API batching."""
    
    def __init__(self, stream_id: str, inference_client: InferenceAPIClient):
        self.stream_id = stream_id
        self.inference_client = inference_client
        self.frame_count = 0
        self.last_drowsiness_score = 0.0  # Will be updated after first batch processing
        self.avg_drowsiness_score = 0.0
        self.max_drowsiness_score = 0.0
        self.alert_history = deque(maxlen=10)
        self.processing_times = deque(maxlen=20)
        self.frame_buffer = deque(maxlen=6)  # Buffer for batch processing (2-3 frames)
        self.batch_size = 2  # Process frames in batches of 2 (optimized for 2 FPS)
        self.face_quality_scores = deque(maxlen=10)  # Track face detection quality
        
        # Logging system for drowsiness scores
        self.score_history = deque(maxlen=300)  # Store scores with timestamps (5 mins worth)
        self.last_log_time = time.time()
        self.log_interval = 1.0  # Log every 1 second
        self.averaging_window = 5.0  # Average over 5 seconds
        self.log_file_path = f"drowsiness_log_{stream_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Initialize log file
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize the drowsiness log file with headers."""
        try:
            with open(self.log_file_path, 'w', newline='') as f:
                f.write("# Vigilant AI Drowsiness Detection Log\n")
                f.write("# Format: timestamp, drowsiness_score (5-second average)\n")
                f.write("# Stream ID: {}\n".format(self.stream_id))
                f.write("# Started: {}\n".format(datetime.now().isoformat()))
                f.write("timestamp,drowsiness_score\n")
            
            logger.info(f"📝 Drowsiness log initialized: {self.log_file_path}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize log file: {e}")
    
    def _log_drowsiness_score(self, score: float, timestamp: float):
        """Log drowsiness score with time-based averaging."""
        try:
            # Add score to history with timestamp
            self.score_history.append((timestamp, score))
            
            # Check if it's time to log (every second)
            current_time = time.time()
            if current_time - self.last_log_time >= self.log_interval:
                
                # Calculate 5-second average
                cutoff_time = current_time - self.averaging_window
                recent_scores = [s for t, s in self.score_history if t >= cutoff_time]
                
                if recent_scores:
                    avg_score = sum(recent_scores) / len(recent_scores)
                    
                    # Format timestamp
                    log_timestamp = datetime.fromtimestamp(current_time).isoformat()
                    
                    # Write to log file
                    with open(self.log_file_path, 'a', newline='') as f:
                        f.write(f"{log_timestamp},{avg_score:.4f}\n")
                    
                    # Update last log time
                    self.last_log_time = current_time
                    
                    logger.debug(f"📝 Logged drowsiness: {log_timestamp}, score={avg_score:.4f} (avg of {len(recent_scores)} scores)")
                    
        except Exception as e:
            logger.error(f"❌ Failed to log drowsiness score: {e}")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        try:
            log_exists = os.path.exists(self.log_file_path)
            log_size = os.path.getsize(self.log_file_path) if log_exists else 0
            
            # Count log entries
            log_entries = 0
            if log_exists:
                with open(self.log_file_path, 'r') as f:
                    log_entries = sum(1 for line in f if not line.startswith('#') and ',' in line) - 1  # Exclude header
            
            return {
                "log_file": self.log_file_path,
                "log_exists": log_exists,
                "log_size_bytes": log_size,
                "log_entries": log_entries,
                "score_history_length": len(self.score_history),
                "last_log_time": datetime.fromtimestamp(self.last_log_time).isoformat(),
                "log_interval_seconds": self.log_interval,
                "averaging_window_seconds": self.averaging_window
            }
        except Exception as e:
            return {
                "log_file": self.log_file_path,
                "error": str(e)
            }
        
    async def process_frame_async(self, frame_base64: str, timestamp: float) -> Optional[DrowsinessAlert]:
        """Process frames using MediaPipe preprocessing and batch inference."""
        start_time = time.time()
        self.frame_count += 1
        
        try:
            # MediaPipe face quality check
            logger.info(f"🔍 MediaPipe face detection for frame {self.frame_count} - {self.stream_id}")
            face_quality = face_detector.detect_face_quality(frame_base64)
            self.face_quality_scores.append(face_quality.get('face_confidence', 0.0))
            
            logger.info(f"   👁️  Face detected: {face_quality.get('face_detected', False)}")
            logger.info(f"   🎯 Face confidence: {face_quality.get('face_confidence', 0.0):.4f}")
            logger.info(f"   📍 Landmark count: {face_quality.get('landmark_count', 0)}")
            if face_quality.get('face_bbox'):
                bbox = face_quality.get('face_bbox')
                logger.info(f"   📦 Face bbox: ({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})")
                logger.info(f"   📏 Face size: {face_quality.get('face_size', 0)} pixels")
            
            # Only process frames with decent face detection
            if not face_quality.get('face_detected', False):
                logger.warning(f"❌ No face detected in frame {self.frame_count} for {self.stream_id}")
                # Return a neutral result instead of None to keep UI updated
                return DrowsinessAlert(
                    stream_id=self.stream_id,
                    timestamp=datetime.now().isoformat(),
                    drowsiness_score=0.5,
                    confidence=0.0,
                    is_drowsy=False,
                    alert_level="no_face",
                    features={"face_detected": False, "face_quality": face_quality},
                    frame_count=self.frame_count
                )
            
            # Add frame to buffer with face quality info
            frame_data = {
                "image_base64": frame_base64,
                "timestamp": timestamp,
                "frame_id": f"{self.stream_id}_frame_{self.frame_count:06d}",
                "stream_id": self.stream_id,
                "face_quality": face_quality
            }
            
            self.frame_buffer.append(frame_data)
            logger.info(f"📥 Frame added to buffer ({len(self.frame_buffer)}/{self.batch_size})")
            logger.info(f"   🏷️  Frame ID: {frame_data['frame_id']}")
            logger.info(f"   ⏰ Timestamp: {timestamp:.3f}")
            logger.info(f"   📊 Face quality: {face_quality.get('face_confidence', 0):.4f}")
            
            # Process batch when buffer is full
            if len(self.frame_buffer) >= self.batch_size:
                logger.info(f"🚀 Buffer full ({self.batch_size} frames) - Starting batch processing")
                return await self._process_batch_with_mediapipe()
            else:
                logger.debug(f"⏳ Waiting for more frames ({len(self.frame_buffer)}/{self.batch_size})")
            
            # Always return current drowsiness score to keep mobile client updated
            # Use the last processed score (even if 0.0) to show real-time updates
            current_score = self.last_drowsiness_score
            
            # Determine alert level based on current score
            if current_score >= 0.7:
                alert_level = "critical"
            elif current_score >= 0.5:
                alert_level = "high"
            elif current_score >= 0.3:
                alert_level = "medium"
            else:
                alert_level = "low"
            
            return DrowsinessAlert(
                stream_id=self.stream_id,
                timestamp=datetime.now().isoformat(),
                drowsiness_score=current_score,
                confidence=face_quality.get('face_confidence', 0.0),
                is_drowsy=current_score > 0.5,
                alert_level=alert_level,
                features={
                    "face_confidence": face_quality.get('face_confidence', 0.0),
                    "buffer_size": len(self.frame_buffer),
                    "batch_size": self.batch_size,
                    "face_detected": True,
                    "processing_status": "buffering"
                },
                frame_count=self.frame_count
            )
                
        except Exception as e:
            logger.error(f"Error processing frame for {self.stream_id}: {e}")
            return DrowsinessAlert(
                stream_id=self.stream_id,
                timestamp=datetime.now().isoformat(),
                drowsiness_score=0.5,
                confidence=0.0,
                is_drowsy=False,
                alert_level="error",
                features={"error": str(e)},
                frame_count=self.frame_count
            )
    
    def add_to_buffer(self, frame_base64: str, timestamp: float):
        """Add frame to buffer for potential batch processing."""
        frame_id = f"{self.stream_id}_buffer_{len(self.frame_buffer):06d}"
        
        self.frame_buffer.append({
            "image_base64": frame_base64,
            "timestamp": timestamp,
            "frame_id": frame_id
        })
    
    async def _process_batch_with_mediapipe(self) -> Optional[DrowsinessAlert]:
        """Process a batch of frames using MediaPipe + video stream API."""
        if not self.frame_buffer:
            return None
            
        start_time = time.time()
        frames = list(self.frame_buffer)
        self.frame_buffer.clear()
        
        try:
            # Check if inference client is available
            if not self.inference_client:
                logger.error(f"❌ Inference client not available for {self.stream_id}")
                return DrowsinessAlert(
                    stream_id=self.stream_id,
                    timestamp=datetime.now().isoformat(),
                    drowsiness_score=0.5,
                    confidence=0.0,
                    is_drowsy=False,
                    alert_level="error",
                    features={"error": "Inference client not available"},
                    frame_count=self.frame_count
                )
            
            # Send batch to video stream API
            logger.info(f"📡 Sending batch of {len(frames)} frames to inference API for {self.stream_id}")
            result = await self.inference_client.batch_infer_frames(frames)
            
            if result and result.get('success'):
                video_result = result.get('data', {})
                
                # Extract video-level statistics
                self.avg_drowsiness_score = video_result.get('avg_drowsiness_score', 0.5)
                self.max_drowsiness_score = video_result.get('max_drowsiness_score', 0.5)
                alert_frames = video_result.get('alert_frames', 0)
                total_frames = video_result.get('total_frames', len(frames))
                
                # Update processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # Use the averaged score for more stable results
                self.last_drowsiness_score = self.avg_drowsiness_score
                
                # Log the drowsiness score (handles time-based averaging and file writing)
                self._log_drowsiness_score(self.avg_drowsiness_score, time.time())
                
                # Send DV data to synchronized inference engine
                if sync_inference_engine is not None:
                    # Send the drowsiness score with timestamp to sync engine
                    current_timestamp = time.time()
                    success = sync_inference_engine.receive_dv_from_inference_api(
                        drowsiness_score=self.avg_drowsiness_score,
                        timestamp=current_timestamp,
                        features={
                            "avg_score": self.avg_drowsiness_score,
                            "max_score": self.max_drowsiness_score,
                            "alert_frames": alert_frames,
                            "total_frames": total_frames,
                            "face_confidence": sum(self.face_quality_scores) / len(self.face_quality_scores) if self.face_quality_scores else 0.0,
                            "batch_size": len(frames),
                            "stream_id": self.stream_id
                        },
                        alert_level=None,  # Will be determined by sync engine
                        confidence=sum(self.face_quality_scores) / len(self.face_quality_scores) if self.face_quality_scores else 0.0,
                        frame_id=f"{self.stream_id}_batch_{self.frame_count}"
                    )
                    
                    if success:
                        logger.info(f"📡 DV data sent to sync engine: score={self.avg_drowsiness_score:.4f}, timestamp={current_timestamp:.3f}")
                    else:
                        logger.warning(f"⚠️  Failed to send DV data to sync engine")
                
                # Determine alert level from average score
                if self.avg_drowsiness_score >= 0.7:
                    alert_level = "critical"
                elif self.avg_drowsiness_score >= 0.5:
                    alert_level = "high"
                elif self.avg_drowsiness_score >= 0.3:
                    alert_level = "medium"
                else:
                    alert_level = "low"
                
                # Calculate confidence from batch processing and face quality
                face_confidence = sum(self.face_quality_scores) / len(self.face_quality_scores) if self.face_quality_scores else 0.0
                batch_confidence = min(1.0, alert_frames / max(total_frames, 1))
                confidence = (face_confidence + batch_confidence) / 2
                
                # Create comprehensive alert
                alert = DrowsinessAlert(
                    stream_id=self.stream_id,
                    timestamp=datetime.now().isoformat(),
                    drowsiness_score=self.avg_drowsiness_score,
                    confidence=confidence,
                    is_drowsy=self.avg_drowsiness_score > 0.5,
                    alert_level=alert_level,
                    features={
                        "avg_score": self.avg_drowsiness_score,
                        "max_score": self.max_drowsiness_score,
                        "alert_frames": alert_frames,
                        "total_frames": total_frames,
                        "face_confidence": face_confidence,
                        "batch_size": len(frames)
                    },
                    frame_count=self.frame_count
                )
                
                self.alert_history.append(alert)
                
                if alert_level in ['high', 'critical']:
                    logger.info(f"🚨 {alert_level.upper()} drowsiness detected for {self.stream_id}: avg={self.avg_drowsiness_score:.3f}")
                else:
                    logger.debug(f"Batch processed for {self.stream_id}: avg={self.avg_drowsiness_score:.3f}")
                
                return alert
                
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No response from inference API'
                logger.error(f"❌ Batch processing failed for {self.stream_id}: {error_msg}")
                
                # Return last known score or neutral default with error status
                fallback_score = self.last_drowsiness_score if hasattr(self, 'last_drowsiness_score') else 0.5
                return DrowsinessAlert(
                    stream_id=self.stream_id,
                    timestamp=datetime.now().isoformat(),
                    drowsiness_score=fallback_score,
                    confidence=0.0,
                    is_drowsy=fallback_score > 0.5,
                    alert_level="error",
                    features={
                        "batch_error": error_msg,
                        "fallback_score": fallback_score,
                        "inference_api_status": "failed"
                    },
                    frame_count=self.frame_count
                )
                
        except Exception as e:
            logger.error(f"Error processing batch for {self.stream_id}: {e}")
            return DrowsinessAlert(
                stream_id=self.stream_id,
                timestamp=datetime.now().isoformat(),
                drowsiness_score=0.5,
                confidence=0.0,
                is_drowsy=False,
                alert_level="error",
                features={"batch_exception": str(e)},
                frame_count=self.frame_count
            )
    
    async def process_buffer_batch(self) -> List[DrowsinessAlert]:
        """Process accumulated frames as a batch (legacy method)."""
        if len(self.frame_buffer) == 0:
            return []
        
        try:
            # Use the new MediaPipe batch processing
            result = await self._process_batch_with_mediapipe()
            return [result] if result else []
            
        except Exception as e:
            logger.error(f"Error in legacy batch processing for {self.stream_id}: {e}")
            return []
            
            if "error" in result:
                logger.error(f"Batch inference error: {result['error']}")
                return []
            
            alerts = []
            frame_results = result.get('frame_results', [])
            
            for frame_result in frame_results:
                # Create alert from batch result
                alert = DrowsinessAlert(
                    stream_id=self.stream_id,
                    timestamp=datetime.now().isoformat(),
                    drowsiness_score=frame_result.get('drowsiness_score', 0.0),
                    confidence=frame_result.get('confidence', 0.0),
                    is_drowsy=frame_result.get('drowsiness_score', 0.0) > 0.5,
                    alert_level=frame_result.get('alert_level', 'low'),
                    features=frame_result.get('features', {}),
                    frame_count=self.frame_count
                )
                
                alerts.append(alert)
                self.alert_history.append(alert)
                self.frame_count += 1
            
            if frame_results:
                self.last_drowsiness_score = frame_results[-1].get('drowsiness_score', 0.0)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error processing buffer batch for stream {self.stream_id}: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        
        return {
            'frames_processed': self.frame_count,
            'avg_processing_time': avg_processing_time,
            'current_drowsiness_score': self.last_drowsiness_score,
            'alerts_triggered': len(self.alert_history),
            'recent_alerts': len([a for a in self.alert_history if a.alert_level in ["high", "critical"]]),
            'buffer_size': len(self.frame_buffer)
        }


@app.on_event("startup")
async def startup_event():
    """Initialize AI components on startup."""
    global camera_manager, inference_client, sync_inference_engine
    
    logger.info("🚀 Starting Vigilant AI Live Stream Service")
    
    try:
        # Initialize camera manager first (always works)
        camera_manager = CameraManager()
        logger.info("✅ Camera manager initialized")
        
        # Initialize synchronized inference engine
        if SynchronizedInferenceEngine is not None:
            sync_inference_engine = SynchronizedInferenceEngine(
                sync_tolerance=0.1,  # 100ms tolerance for synchronization
                max_buffer_size=1000,
                sync_timeout=2.0,    # 2 second timeout for missing data
                enable_logging=True
            )
            logger.info("✅ Synchronized Inference Engine initialized")
        else:
            logger.warning("⚠️  Synchronized Inference Engine not available")
        
        # Initialize inference API client (non-blocking)
        logger.info("Initializing Inference API client...")
        inference_client = InferenceAPIClient(INFERENCE_API_URL)
        
        # Start background task to connect to inference API
        asyncio.create_task(connect_to_inference_api())
        
        # Start cleanup task
        asyncio.create_task(cleanup_streams())
        
        logger.info("✅ Live Stream Service startup completed")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        logger.error(traceback.format_exc())
        # Don't raise - allow service to start even if some components fail
        
        # Initialize minimal components
        if camera_manager is None:
            camera_manager = CameraManager()


async def connect_to_inference_api():
    """Background task to connect to inference API with retries."""
    global inference_client
    
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to connect to Inference API (attempt {attempt + 1}/{max_retries})...")
            
            if await inference_client.initialize():
                logger.info("✅ Connected to Inference API successfully")
                return
            else:
                logger.warning(f"⚠️  Could not connect to Inference API - attempt {attempt + 1} failed")
                
        except Exception as e:
            logger.error(f"❌ Inference API connection attempt {attempt + 1} failed: {e}")
        
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
    
    logger.warning("⚠️  Could not connect to Inference API after all attempts - service will run with limited functionality")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global inference_client
    
    logger.info("🛑 Shutting down Live Stream Service")
    
    if inference_client:
        await inference_client.close()
        logger.info("✅ Inference client closed")


async def cleanup_streams():
    """Background task to cleanup inactive streams."""
    while True:
        try:
            current_time = time.time()
            streams_to_remove = []
            
            for stream_id, stream_data in active_streams.items():
                if current_time - stream_data.get('last_activity', 0) > MAX_STREAM_DURATION:
                    streams_to_remove.append(stream_id)
            
            for stream_id in streams_to_remove:
                await stop_stream(stream_id)
                logger.info(f"🧹 Cleaned up inactive stream: {stream_id}")
            
            await asyncio.sleep(CLEANUP_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(CLEANUP_INTERVAL)


async def stop_stream(stream_id: str):
    """Stop and cleanup a stream."""
    if stream_id in active_streams:
        # Release camera
        if camera_manager:
            camera_manager.release_camera(stream_id)
        
        # Remove from active streams
        del active_streams[stream_id]
        
        logger.info(f"🛑 Stream {stream_id} stopped and cleaned up")




@app.websocket("/ws/mobile_stream")
async def websocket_mobile_stream(websocket: WebSocket):
    """WebSocket endpoint for mobile camera stream processing."""
    await websocket.accept()
    stream_id = str(uuid.uuid4())
    
    logger.info(f"📱 New Mobile WebSocket connection: {stream_id}")
    
    try:
        # Wait for mobile configuration
        config_data = await websocket.receive_text()
        config = json.loads(config_data)
        
        if config.get('type') != 'mobile_config':
            await websocket.send_text(json.dumps({
                "error": "Expected mobile_config message"
            }))
            return
        
        stream_id = config.get('stream_id', stream_id)
        frame_rate = config.get('frame_rate', 5)
        device_info = config.get('device_info', {})
        
        # Create mobile frame processor with inference client
        frame_processor = StreamFrameProcessor(stream_id, inference_client)
        
        # Add to active streams
        active_streams[stream_id] = {
            'start_time': datetime.now().isoformat(),
            'last_activity': time.time(),
            'websocket': websocket,
            'config': config,
            'status': 'active',
            'type': 'mobile',
            'device_info': device_info,
            'frame_processor': frame_processor
        }
        
        # Send configuration acknowledgment
        await websocket.send_text(json.dumps({
            "type": "config_ack",
            "stream_id": stream_id,
            "status": "Mobile stream configured successfully",
            "expected_fps": frame_rate
        }))
        
        logger.info(f"📱 Mobile stream configured: {stream_id} at {frame_rate} FPS")
        
        # Main message processing loop
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get('type') == 'frame':
                    # Process mobile frame
                    await process_mobile_frame(stream_id, data, frame_processor, websocket)
                    
                    # Update last activity
                    active_streams[stream_id]['last_activity'] = time.time()
                
                elif data.get('type') == 'ping':
                    # Respond to ping
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": time.time()
                    }))
                
                else:
                    logger.warning(f"Unknown message type from mobile: {data.get('type')}")
            
            except asyncio.TimeoutError:
                # Check if stream is still alive
                if stream_id not in active_streams:
                    break
                continue
    
    except WebSocketDisconnect:
        logger.info(f"📱 Mobile WebSocket disconnected: {stream_id}")
    except Exception as e:
        logger.error(f"❌ Mobile WebSocket error for stream {stream_id}: {e}")
        try:
            await websocket.send_text(json.dumps({
                "error": f"Mobile stream error: {str(e)}"
            }))
        except:
            pass
    
    finally:
        # Cleanup
        await stop_stream(stream_id)


async def process_mobile_frame(stream_id: str, frame_data: dict, frame_processor: StreamFrameProcessor, websocket: WebSocket):
    """Process a frame received from mobile device using Inference API."""
    try:
        # Get base64 image data
        image_base64 = frame_data.get('image_base64', '')
        if not image_base64:
            return
        
        # Remove data URL prefix if present
        if image_base64.startswith('data:image'):
            image_base64 = image_base64.split(',')[1]
        
        timestamp = frame_data.get('timestamp', time.time())
        
        # Process frame through inference API
        alert = await frame_processor.process_frame_async(image_base64, timestamp)
        
        # Send alert if generated
        if alert is not None:
            await websocket.send_text(alert.json())
            
            # Log significant alerts
            if alert.alert_level in ['high', 'critical']:
                logger.warning(f"📱 Mobile Alert [{alert.alert_level.upper()}]: "
                             f"Score={alert.drowsiness_score:.3f}, Stream={stream_id}")
        
        # Send periodic status updates
        if frame_processor.frame_count % 25 == 0:  # Every 5 seconds at 5fps
            stats = frame_processor.get_stats()
            await websocket.send_text(json.dumps({
                "type": "status_update",
                "stream_id": stream_id,
                "frames_processed": stats['frames_processed'],
                "current_score": stats['current_drowsiness_score'],
                "avg_processing_time": stats['avg_processing_time']
            }))
    
    except Exception as e:
        logger.error(f"Error processing mobile frame for {stream_id}: {e}")
        await websocket.send_text(json.dumps({
            "error": f"Frame processing error: {str(e)}"
        }))


@app.post("/start_camera_stream", response_model=Dict[str, str])
async def start_camera_stream(config: StreamConfig):
    """Start a new camera stream."""
    try:
        stream_id = config.stream_id
        
        # Initialize camera
        if not camera_manager.initialize_camera(
            stream_id, config.camera_source, config.frame_width, config.frame_height
        ):
            raise HTTPException(status_code=400, detail="Failed to initialize camera")
        
        # Add to active streams
        active_streams[stream_id] = {
            'start_time': datetime.now().isoformat(),
            'last_activity': time.time(),
            'config': config,
            'status': 'active'
        }
        
        # Start background processing
        asyncio.create_task(process_camera_stream(stream_id))
        
        return {
            "stream_id": stream_id,
            "status": "started",
            "message": "Camera stream started successfully"
        }
        
    except Exception as e:
        logger.error(f"Error starting camera stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_camera_stream(stream_id: str):
    """Background task to process camera stream."""
    try:
        frame_processor = camera_manager.frame_processors[stream_id]
        
        while stream_id in active_streams and active_streams[stream_id]['status'] == 'active':
            frame = camera_manager.get_frame(stream_id)
            
            if frame is not None:
                alert = frame_processor.process_frame(frame)
                
                if alert is not None:
                    # Store latest alert
                    active_streams[stream_id]['latest_alert'] = alert.dict()
                
                # Update activity
                active_streams[stream_id]['last_activity'] = time.time()
            
            await asyncio.sleep(0.2)  # 5fps processing
            
    except Exception as e:
        logger.error(f"Error processing camera stream {stream_id}: {e}")
        if stream_id in active_streams:
            active_streams[stream_id]['status'] = 'error'


@app.get("/stream_status/{stream_id}", response_model=StreamStatus)
async def get_stream_status(stream_id: str):
    """Get status of a specific stream."""
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    stream_data = active_streams[stream_id]
    frame_processor = camera_manager.frame_processors.get(stream_id)
    
    stats = frame_processor.get_stats() if frame_processor else {}
    
    return StreamStatus(
        stream_id=stream_id,
        status=stream_data['status'],
        start_time=stream_data['start_time'],
        frames_processed=stats.get('frames_processed', 0),
        alerts_triggered=stats.get('alerts_triggered', 0),
        current_drowsiness_score=stats.get('current_drowsiness_score', 0.0),
        avg_processing_time=stats.get('avg_processing_time', 0.0)
    )


@app.get("/latest_alert/{stream_id}")
async def get_latest_alert(stream_id: str):
    """Get the latest drowsiness alert for a stream."""
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    latest_alert = active_streams[stream_id].get('latest_alert')
    
    if latest_alert is None:
        return {"message": "No alerts yet", "stream_id": stream_id}
    
    return latest_alert


@app.delete("/stop_stream/{stream_id}")
async def stop_camera_stream(stream_id: str):
    """Stop a camera stream."""
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    await stop_stream(stream_id)
    
    return {"message": f"Stream {stream_id} stopped successfully"}


@app.get("/active_streams")
async def get_active_streams():
    """Get all active streams."""
    streams = []
    for stream_id, stream_data in active_streams.items():
        streams.append({
            "stream_id": stream_id,
            "status": stream_data['status'],
            "start_time": stream_data['start_time'],
            "camera_source": stream_data.get('config', {}).camera_source if hasattr(stream_data.get('config', {}), 'camera_source') else 'unknown'
        })
    
    return {
        "active_streams": len(streams),
        "streams": streams
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Test inference API connection
    inference_api_healthy = False
    if inference_client:
        try:
            async with inference_client.session.get(f"{INFERENCE_API_URL}/inference/health") as response:
                if response.status == 200:
                    inference_api_healthy = True
        except:
            inference_api_healthy = False
    
    # Get logging stats from active streams
    logging_stats = {}
    for stream_id, stream_info in active_streams.items():
        # Check if stream has processor with logging capability
        processor = None
        for manager in camera_manager.frame_processors.values():
            if hasattr(manager, 'stream_id') and manager.stream_id == stream_id:
                processor = manager
                break
        
        if processor and hasattr(processor, 'get_log_stats'):
            logging_stats[stream_id] = processor.get_log_stats()
    
    return {
        "status": "healthy",
        "service": "Live Stream Drowsiness Detection",
        "version": "2.0.0",
        "components": {
            "camera_manager": camera_manager is not None,
            "inference_client": inference_client is not None,
            "inference_api_connected": inference_api_healthy,
            "sync_inference_engine": sync_inference_engine is not None
        },
        "active_streams": len(active_streams),
        "inference_api_url": INFERENCE_API_URL,
        "uptime_seconds": time.time() - service_start_time if 'service_start_time' in globals() else 0,
        "logging": {
            "enabled": True,
            "active_logs": len(logging_stats),
            "streams": logging_stats
        }
    }


@app.get("/logs/{stream_id}")
async def get_stream_logs(stream_id: str):
    """Get drowsiness logs for a specific stream."""
    # Find the processor for this stream
    processor = None
    for manager in camera_manager.frame_processors.values():
        if hasattr(manager, 'stream_id') and manager.stream_id == stream_id:
            processor = manager
            break
    
    if not processor:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    if not hasattr(processor, 'get_log_stats'):
        raise HTTPException(status_code=400, detail="Logging not available for this stream")
    
    log_stats = processor.get_log_stats()
    
    # Read recent log entries (last 50)
    log_entries = []
    if log_stats.get('log_exists', False):
        try:
            with open(log_stats['log_file'], 'r') as f:
                lines = f.readlines()
                # Skip comments and header, get last 50 data lines
                data_lines = [line.strip() for line in lines if not line.startswith('#') and ',' in line][1:]  # Skip header
                recent_entries = data_lines[-50:] if len(data_lines) > 50 else data_lines
                
                for entry in recent_entries:
                    if ',' in entry:
                        timestamp, score = entry.split(',', 1)
                        log_entries.append({
                            "timestamp": timestamp,
                            "drowsiness_score": float(score)
                        })
        except Exception as e:
            log_entries = [{"error": f"Failed to read log file: {e}"}]
    
    return {
        "stream_id": stream_id,
        "log_stats": log_stats,
        "recent_entries": log_entries,
        "entry_count": len(log_entries)
    }


@app.get("/logs")
async def get_all_logs():
    """Get logging information for all active streams."""
    all_logs = {}
    
    for manager in camera_manager.frame_processors.values():
        if hasattr(manager, 'get_log_stats') and hasattr(manager, 'stream_id'):
            all_logs[manager.stream_id] = manager.get_log_stats()
    
    return {
        "active_processors": len(camera_manager.frame_processors),
        "logged_streams": len(all_logs),
        "logs": all_logs
    }


class HVDataRequest(BaseModel):
    hv_data: dict
    timestamp: float
    source: str = "vitals_api"

@app.post("/sync_engine/receive_hv_data")
async def receive_hv_data_endpoint(request: HVDataRequest):
    """
    Receive HV data from external sources (like vitals processor) and send to sync engine.
    
    Args:
        request: HVDataRequest containing hv_data, timestamp, and source
    """
    if sync_inference_engine is None:
        raise HTTPException(status_code=503, detail="Synchronized Inference Engine not available")
    
    try:
        success = sync_inference_engine.receive_hv_data(request.hv_data, request.timestamp, request.source)
        return {
            "success": success,
            "message": "HV data received" if success else "Failed to process HV data",
            "timestamp": request.timestamp,
            "source": request.source
        }
    except Exception as e:
        logger.error(f"Error receiving HV data via API: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing HV data: {e}")


@app.get("/sync_engine_status")
async def get_sync_engine_status():
    """Get status of the synchronized inference engine."""
    if sync_inference_engine is None:
        return {
            "status": "not_available",
            "message": "Synchronized Inference Engine not initialized"
        }
    
    try:
        # Get buffer status and statistics
        buffer_status = sync_inference_engine.get_buffer_status()
        
        # Get latest result
        latest_result = sync_inference_engine.get_latest_result()
        
        # Get health score trends
        trends = sync_inference_engine.get_health_score_trends()
        
        # Get rule check status
        rule_status = sync_inference_engine.get_rule_check_status()
        
        return {
            "status": "active",
            "buffer_status": buffer_status,
            "latest_result": {
                "timestamp": latest_result.timestamp if latest_result else None,
                "dv_score": latest_result.dv_score if latest_result else None,
                "hv_score": latest_result.hv_score if latest_result else None,
                "health_score": latest_result.health_score if latest_result else None,
                "mode": latest_result.mode if latest_result else None
            } if latest_result else None,
            "trends": trends,
            "rule_checks": rule_status,
            "sync_config": {
                "sync_tolerance": sync_inference_engine.sync_tolerance,
                "max_buffer_size": sync_inference_engine.max_buffer_size,
                "sync_timeout": sync_inference_engine.sync_timeout
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting sync engine status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Vigilant AI - Live Stream Drowsiness Detection",
        "version": "2.0.0",
        "status": "running",
        "description": "Real-time drowsiness detection from live camera streams",
        "endpoints": {
            "mobile_websocket": "/ws/mobile_stream",
            "start_stream": "/start_camera_stream",
            "stream_status": "/stream_status/{stream_id}",
            "latest_alert": "/latest_alert/{stream_id}",
            "stop_stream": "/stop_stream/{stream_id}",
            "active_streams": "/active_streams",
            "sync_engine_status": "/sync_engine_status",
            "health": "/health",
            "docs": "/docs"
        }
    }


# Global service start time
service_start_time = time.time()


def create_self_signed_cert():
    """Create self-signed certificate for HTTPS/WSS."""
    import subprocess
    import os
    
    cert_file = "server.crt"
    key_file = "server.key"
    
    if os.path.exists(cert_file) and os.path.exists(key_file):
        print("✅ Certificate files already exist")
        return cert_file, key_file
    
    print("🔒 Creating self-signed certificate for HTTPS/WSS...")
    
    try:
        # Create self-signed certificate
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:4096", 
            "-keyout", key_file, "-out", cert_file, "-days", "365", "-nodes",
            "-subj", "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        ], check=True, capture_output=True)
        
        print(f"✅ Created certificate: {cert_file}")
        print(f"✅ Created private key: {key_file}")
        return cert_file, key_file
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create certificate: {e}")
        print("💡 Install OpenSSL: sudo apt-get install openssl")
        return None, None
    except FileNotFoundError:
        print("❌ OpenSSL not found. Install it first:")
        print("   sudo apt-get install openssl")
        return None, None


if __name__ == "__main__":
    print("🚀 Starting Live Stream Service on port 8001...")
    print("📡 WebSocket will be available via HTTPS")
    
    # Run the server with SSL
    uvicorn.run(
        "live_stream_service:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
        ssl_keyfile="server.key",
        ssl_certfile="server.crt"
    )

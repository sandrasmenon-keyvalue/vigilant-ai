#!/usr/bin/env python3
"""
Integrated Live Stream Service
End-to-end integration connecting WebSocket, inference_api, synchronized_inference_engine,
and vitals WebSocket for unified processing and alerting.
"""

import asyncio
import base64
import cv2
import os
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import deque

import aiohttp
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our integrated components
from inference.synchronized_inference_engine import SynchronizedInferenceEngine
from vitals_processing.vitals_websocket_processor import VitalsWebSocketProcessor
from actions.alert_prediction.alert_prediction import Alert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
INFERENCE_API_URL = "http://localhost:8002"
VITALS_WEBSOCKET_URL = os.getenv("VITALS_WEBSOCKET_URI", "ws://192.168.5.93:4000/ws?token=dev-shared-secret")
VITALS_API_KEY = os.getenv("VITALS_API_KEY")
VITALS_AUTH_TOKEN = os.getenv("VITALS_AUTH_TOKEN")

# Global instances
synchronized_engine = None
vitals_processor = None
active_streams = {}


@dataclass
class StreamConfig:
    """Stream configuration."""
    stream_id: str
    camera_source: int = 0
    frame_width: int = 640
    frame_height: int = 480
    enable_display: bool = True


@dataclass
class IntegratedAlert:
    """Integrated alert containing all alert types."""
    stream_id: str
    timestamp: str
    alert_type: str
    alert_level: str
    reason: str
    drowsiness_score: float = 0.0
    health_score: float = 0.0
    environmental_conditions: Dict[str, Any] = None
    vitals_data: Dict[str, Any] = None


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
                    result = await response.json()
                    return result
                else:
                    logger.error(f"❌ Single frame inference failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Error in single frame inference: {e}")
            return None


class IntegratedStreamProcessor:
    """Processes frames with integrated DV, HV, and environmental monitoring."""
    
    def __init__(self, stream_id: str, inference_client: InferenceAPIClient):
        self.stream_id = stream_id
        self.inference_client = inference_client
        self.frame_count = 0
        self.last_drowsiness_score = 0.0
        self.alert_history = deque(maxlen=10)
        self.processing_times = deque(maxlen=20)
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'alerts_triggered': 0,
            'avg_processing_time': 0.0
        }
    
    async def process_frame(self, frame: cv2.Mat) -> Optional[IntegratedAlert]:
        """Process a single frame with integrated monitoring."""
        start_time = time.time()
        self.frame_count += 1
        
        try:
            # Encode frame to base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Get current timestamp
            current_timestamp = time.time()
            frame_id = f"{self.stream_id}_frame_{self.frame_count:06d}"
            
            # Call inference API for DV data
            inference_result = await self.inference_client.infer_single_frame(
                image_base64=frame_base64,
                timestamp=current_timestamp,
                frame_id=frame_id
            )
            
            if not inference_result:
                logger.warning(f"⚠️ No inference result for frame {self.frame_count}")
                return None
            
            # Extract DV data from inference result
            drowsiness_score = inference_result.get('drowsiness_score', 0.0)
            alert_level = inference_result.get('alert_level', 'low')
            confidence = inference_result.get('confidence', 0.0)
            features = inference_result.get('features', {})
            
            # Update drowsiness score
            self.last_drowsiness_score = drowsiness_score
            
            # Send DV data to synchronized inference engine
            if synchronized_engine:
                success = synchronized_engine.receive_dv_from_inference_api(
                    drowsiness_score=drowsiness_score,
                    timestamp=current_timestamp,
                    features=features,
                    alert_level=alert_level,
                    confidence=confidence,
                    frame_id=frame_id
                )
                
                if success:
                    logger.debug(f"📹 DV data sent to synchronized engine: score={drowsiness_score:.3f}")
                else:
                    logger.warning(f"⚠️ Failed to send DV data to synchronized engine")
            
            # Check if we should trigger an alert based on drowsiness
            if drowsiness_score >= 0.7:
                alert_level = "critical"
            elif drowsiness_score >= 0.5:
                alert_level = "high"
            elif drowsiness_score >= 0.3:
                alert_level = "medium"
            else:
                alert_level = "low"
            
            # Create integrated alert if drowsiness is concerning
            if drowsiness_score >= 0.5:  # Trigger alert for medium+ drowsiness
                alert = IntegratedAlert(
                    stream_id=self.stream_id,
                    timestamp=datetime.now().isoformat(),
                    alert_type="drowsiness",
                    alert_level=alert_level,
                    reason=f"Driver appears drowsy (score: {drowsiness_score:.3f})",
                    drowsiness_score=drowsiness_score
                )
                
                self.alert_history.append(alert)
                self.stats['alerts_triggered'] += 1
                
                logger.warning(f"🚨 Drowsiness alert: {alert_level} - {drowsiness_score:.3f}")
                return alert
            
            # Update processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.stats['frames_processed'] += 1
            self.stats['avg_processing_time'] = sum(self.processing_times) / len(self.processing_times)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error processing frame: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.stats,
            'frame_count': self.frame_count,
            'last_drowsiness_score': self.last_drowsiness_score,
            'alert_history_count': len(self.alert_history)
        }


class IntegratedCameraManager:
    """Manages camera streams with integrated processing."""
    
    def __init__(self):
        self.cameras = {}
        self.frame_processors = {}
        self.inference_client = None
    
    def initialize_camera(self, stream_id: str, camera_source: int, width: int, height: int) -> bool:
        """Initialize camera for a stream."""
        try:
            # Initialize camera
            cap = cv2.VideoCapture(camera_source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            if not cap.isOpened():
                logger.error(f"❌ Failed to open camera {camera_source}")
                return False
            
            self.cameras[stream_id] = cap
            
            # Initialize frame processor
            if not self.inference_client:
                self.inference_client = InferenceAPIClient(INFERENCE_API_URL)
            
            self.frame_processors[stream_id] = IntegratedStreamProcessor(
                stream_id, self.inference_client
            )
            
            logger.info(f"✅ Camera {camera_source} initialized for stream {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing camera: {e}")
            return False
    
    def get_frame(self, stream_id: str) -> Optional[cv2.Mat]:
        """Get frame from camera."""
        if stream_id not in self.cameras:
            return None
        
        cap = self.cameras[stream_id]
        ret, frame = cap.read()
        
        if ret:
            return frame
        else:
            logger.warning(f"⚠️ Failed to read frame from camera {stream_id}")
            return None
    
    def cleanup_stream(self, stream_id: str):
        """Cleanup camera and processor for a stream."""
        if stream_id in self.cameras:
            self.cameras[stream_id].release()
            del self.cameras[stream_id]
        
        if stream_id in self.frame_processors:
            del self.frame_processors[stream_id]
        
        logger.info(f"🧹 Cleaned up stream {stream_id}")


# Initialize FastAPI app
app = FastAPI(title="Integrated Live Stream Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
camera_manager = None


async def handle_health_score(result):
    """Handle health score results from synchronized inference."""
    logger.info(f"🏥 Health Score: DV={result.dv_score:.3f}, HV={result.hv_score:.3f}, "
               f"Health={result.health_score:.3f}")


async def handle_alert(alert: Alert, alert_type: str):
    """Handle alerts from synchronized inference engine."""
    logger.warning(f"🚨 Alert [{alert_type}]: {alert.reason}")
    
    # Broadcast alert to all active streams
    for stream_id, stream_data in active_streams.items():
        if 'websocket' in stream_data:
            try:
                alert_data = {
                    "type": "integrated_alert",
                    "alert_type": alert_type,
                    "alert_level": alert.type,
                    "reason": alert.reason,
                    "timestamp": datetime.now().isoformat(),
                    "stream_id": stream_id
                }
                await stream_data['websocket'].send_text(json.dumps(alert_data))
            except Exception as e:
                logger.error(f"❌ Failed to send alert to stream {stream_id}: {e}")


async def handle_vitals_data(vitals_data: Dict[str, Any]):
    """Handle vitals data from WebSocket."""
    logger.debug(f"📊 Vitals received: HR={vitals_data.get('hr')}, SpO2={vitals_data.get('spo2')}")
    
    # Send to synchronized inference engine
    if synchronized_engine:
        current_timestamp = time.time()
        success = synchronized_engine.receive_hv_data(
            hv_data=vitals_data,
            timestamp=current_timestamp,
            source="vitals_websocket"
        )
        
        if success:
            logger.debug(f"📊 HV data sent to synchronized engine")
        else:
            logger.warning(f"⚠️ Failed to send HV data to synchronized engine")


@app.on_event("startup")
async def startup_event():
    """Initialize integrated components on startup."""
    global synchronized_engine, vitals_processor, camera_manager
    
    logger.info("🚀 Starting Integrated Live Stream Service")
    
    try:
        # Initialize synchronized inference engine
        logger.info("🧠 Initializing Synchronized Inference Engine...")
        synchronized_engine = SynchronizedInferenceEngine(
            sync_tolerance=0.1,
            sync_timeout=2.0,  # Wait 2 seconds for missing data type
            health_score_callback=handle_health_score,
            alert_callback=handle_alert,
            enable_logging=True
        )
        logger.info("✅ Synchronized Inference Engine initialized")
        
        # Initialize vitals WebSocket processor
        logger.info("💓 Initializing Vitals WebSocket Processor...")
        vitals_processor = VitalsWebSocketProcessor(
            websocket_uri=VITALS_WEBSOCKET_URL,
            api_key=VITALS_API_KEY,
            auth_token=VITALS_AUTH_TOKEN,
            age=35,  # Default age, should be configurable
            health_conditions={
                'diabetes': 0,
                'hypertension': 0,
                'heart_disease': 0,
                'respiratory_condition': 0,
                'smoker': 0
            }
        )
        
        # Set up vitals callbacks
        vitals_processor.set_raw_vitals_callback(handle_vitals_data)
        
        # Start vitals processor in background
        asyncio.create_task(vitals_processor.start())
        logger.info("✅ Vitals WebSocket Processor started")
        
        # Initialize camera manager
        camera_manager = IntegratedCameraManager()
        logger.info("✅ Camera Manager initialized")
        
        logger.info("🎉 All integrated components initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize integrated components: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global synchronized_engine, vitals_processor
    
    logger.info("🛑 Shutting down integrated services...")
    
    if vitals_processor:
        await vitals_processor.stop()
    
    if camera_manager and camera_manager.inference_client:
        await camera_manager.inference_client.close()
    
    logger.info("✅ Shutdown complete")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "integrated_live_stream",
        "timestamp": datetime.now().isoformat(),
        "active_streams": len(active_streams),
        "synchronized_engine": synchronized_engine is not None,
        "vitals_processor": vitals_processor is not None
    }


@app.websocket("/ws/integrated_stream")
async def websocket_integrated_stream(websocket: WebSocket):
    """WebSocket endpoint for integrated video stream processing."""
    await websocket.accept()
    stream_id = str(uuid.uuid4())
    
    logger.info(f"🔗 New integrated WebSocket connection: {stream_id}")
    
    try:
        # Wait for stream configuration
        config_data = await websocket.receive_text()
        config = StreamConfig.parse_raw(config_data)
        stream_id = config.stream_id
        
        # Initialize camera
        if not camera_manager.initialize_camera(
            stream_id, config.camera_source, config.frame_width, config.frame_height
        ):
            await websocket.send_text(json.dumps({
                "error": "Failed to initialize camera"
            }))
            return
        
        # Add to active streams
        active_streams[stream_id] = {
            'start_time': datetime.now().isoformat(),
            'last_activity': time.time(),
            'websocket': websocket,
            'config': config,
            'status': 'active'
        }
        
        # Send confirmation
        await websocket.send_text(json.dumps({
            "status": "integrated_stream_started",
            "stream_id": stream_id,
            "message": "Integrated stream processing started (DV + HV + Environmental)"
        }))
        
        # Main processing loop
        frame_processor = camera_manager.frame_processors[stream_id]
        
        while True:
            # Get frame from camera
            frame = camera_manager.get_frame(stream_id)
            
            if frame is not None:
                # Process frame with integrated monitoring
                alert = await frame_processor.process_frame(frame)
                
                # Send alert if generated
                if alert is not None:
                    await websocket.send_text(json.dumps({
                        "type": "integrated_alert",
                        "alert_type": alert.alert_type,
                        "alert_level": alert.alert_level,
                        "reason": alert.reason,
                        "drowsiness_score": alert.drowsiness_score,
                        "timestamp": alert.timestamp,
                        "stream_id": stream_id
                    }))
                
                # Update last activity
                active_streams[stream_id]['last_activity'] = time.time()
                
                # Optional: Send frame for display
                if config.enable_display and frame_processor.frame_count % 15 == 0:
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    await websocket.send_text(json.dumps({
                        "type": "frame",
                        "frame_data": frame_b64,
                        "stream_id": stream_id
                    }))
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)
    
    except WebSocketDisconnect:
        logger.info(f"🔌 Integrated WebSocket disconnected: {stream_id}")
    except Exception as e:
        logger.error(f"❌ Integrated WebSocket error for stream {stream_id}: {e}")
        await websocket.send_text(json.dumps({
            "error": f"Stream error: {str(e)}"
        }))
    
    finally:
        # Cleanup
        if stream_id in active_streams:
            del active_streams[stream_id]
        
        camera_manager.cleanup_stream(stream_id)
        logger.info(f"🧹 Cleaned up integrated stream {stream_id}")


if __name__ == "__main__":
    print("🚗 Vigilant AI - Integrated Live Stream Service")
    print("=" * 60)
    print("🔗 WebSocket: ws://localhost:8001/ws/integrated_stream")
    print("📊 Health Check: http://localhost:8001/health")
    print("🧠 Inference API: http://localhost:8002")
    print("💓 Vitals WebSocket: ws://localhost:8765")
    print("=" * 60)
    
    uvicorn.run(
        "integrated_live_stream_service:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )

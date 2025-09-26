"""
Vigilant AI Inference API
Dedicated API for drowsiness detection inference from video streams.
"""

import io
import os
import sys
import time
import base64
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import traceback

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add ai_module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_module'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_module', 'vision_model'))

# Import synchronized inference engine
try:
    from inference.synchronized_inference_engine import SynchronizedInferenceEngine
except ImportError as e:
    logging.error(f"Failed to import synchronized inference engine: {e}")
    SynchronizedInferenceEngine = None

# Import AI modules
try:
    from ai_module.vision_model.face_landmarker_live import FaceLandmarkDetector  # Keep for compatibility
    from ai_module.vision_model.face_landmarker_live import LiveFaceLandmarkerDetector, FaceLandmarkDetector as NewFaceLandmarkDetector
    # from feature_extraction import DrowsinessFeatureExtractor  # OLD
    # from enhanced_feature_extraction import EnhancedDrowsinessFeatureExtractor as DrowsinessFeatureExtractor  # ENHANCED
    from ai_module.vision_model.drowsiness_optimized_extraction import DrowsinessOptimizedExtractor as DrowsinessFeatureExtractor  # OPTIMIZED - Stronger signals!
    from ai_module.vision_model.window_processing import SlidingWindowProcessor
    import pickle
    import joblib
except ImportError as e:
    logging.error(f"Failed to import AI modules: {e}")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Vigilant AI - Inference API",
    description="Drowsiness detection inference from video streams",
    version="3.0.0",
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

# Global AI components
face_detector: Optional[FaceLandmarkDetector] = None
feature_extractor: Optional[DrowsinessFeatureExtractor] = None
window_processor: Optional[SlidingWindowProcessor] = None
drowsiness_model = None
feature_scaler = None
sync_inference_engine = None


# Pydantic models
class FrameData(BaseModel):
    """Single frame data for inference."""
    image_base64: str = Field(..., description="Base64 encoded image")
    timestamp: Optional[float] = Field(default=None, description="Frame timestamp")
    frame_id: Optional[str] = Field(default=None, description="Frame identifier")


class VideoStreamInference(BaseModel):
    """Video stream inference request."""
    frames: List[FrameData] = Field(..., description="List of video frames")
    processing_options: Optional[Dict[str, Any]] = Field(default={}, description="Processing options")


class InferenceResult(BaseModel):
    """Single frame inference result."""
    frame_id: Optional[str] = Field(default=None, description="Frame identifier")
    timestamp: float = Field(..., description="Processing timestamp")
    face_detected: bool = Field(..., description="Whether a face was detected")
    features: Dict[str, float] = Field(..., description="Extracted features")
    drowsiness_score: float = Field(..., description="Drowsiness probability (0-1)")
    alert_level: str = Field(..., description="Alert level (low/medium/high/critical)")
    confidence: float = Field(..., description="Prediction confidence")


class VideoInferenceResponse(BaseModel):
    """Video inference response."""
    total_frames: int = Field(..., description="Total frames processed")
    processed_frames: int = Field(..., description="Successfully processed frames")
    processing_time: float = Field(..., description="Total processing time in seconds")
    avg_drowsiness_score: float = Field(..., description="Average drowsiness score")
    max_drowsiness_score: float = Field(..., description="Maximum drowsiness score")
    alert_frames: int = Field(..., description="Number of frames with high/critical alerts")
    frame_results: List[InferenceResult] = Field(..., description="Per-frame results")
    summary: Dict[str, Any] = Field(..., description="Processing summary")


class BatchInferenceRequest(BaseModel):
    """Batch inference request with multiple videos."""
    videos: List[VideoStreamInference] = Field(..., description="List of video streams")
    batch_id: Optional[str] = Field(default=None, description="Batch identifier")


@app.on_event("startup")
async def startup_event():
    """Initialize AI components on startup."""
    global face_detector, feature_extractor, window_processor, drowsiness_model, feature_scaler, sync_inference_engine
    
    logger.info("🚀 Starting Vigilant AI Inference Service")
    
    try:
        # Initialize AI components
        logger.info("Loading AI modules...")
        # Initialize advanced face detector with MediaPipe Face Landmarker  
        logger.info("🚀 Initializing MediaPipe Face Landmarker (478 landmarks, live streaming optimized)")
        face_detector = NewFaceLandmarkDetector(
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            num_faces=1,
            output_blendshapes=True  # Enable facial expressions for enhanced drowsiness detection
        )
        
        feature_extractor = DrowsinessFeatureExtractor(
            ear_threshold=0.25,
            mar_threshold=0.5
        )
        
        window_processor = SlidingWindowProcessor(
            window_size_seconds=5.0,
            fps=5.0
        )
        
        
        # Load trained model and scaler
        model_path = "trained_models/enhanced_vision_training_limited/models/drowsiness_model.pkl"
        scaler_path = "trained_models/enhanced_vision_training_limited/models/feature_scaler.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            # Load the trained model (try joblib first, then pickle)
            try:
                drowsiness_model = joblib.load(model_path)
                logger.info(f"✅ Loaded trained model from {model_path} (joblib)")
            except Exception as e:
                logger.warning(f"Joblib failed, trying pickle: {e}")
                with open(model_path, 'rb') as f:
                    drowsiness_model = pickle.load(f)
                logger.info(f"✅ Loaded trained model from {model_path} (pickle)")
            
            # Load the feature scaler (try joblib first, then pickle)
            try:
                feature_scaler = joblib.load(scaler_path)
                logger.info(f"✅ Loaded feature scaler from {scaler_path} (joblib)")
            except Exception as e:
                logger.warning(f"Joblib failed, trying pickle: {e}")
                with open(scaler_path, 'rb') as f:
                    feature_scaler = pickle.load(f)
                logger.info(f"✅ Loaded feature scaler from {scaler_path} (pickle)")
        else:
            logger.warning(f"⚠️  Model files not found: {model_path} or {scaler_path}")
            drowsiness_model = None
            feature_scaler = None
        
        # Initialize synchronized inference engine (optional)
        if SynchronizedInferenceEngine is not None:
            sync_inference_engine = SynchronizedInferenceEngine(
                sync_tolerance=0.1,  # 100ms tolerance for synchronization
                max_buffer_size=1000,
                sync_timeout=2.0,    # 2 second timeout for missing data
                enable_logging=True
            )
            logger.info("✅ Synchronized Inference Engine initialized in Inference API")
        else:
            logger.warning("⚠️  Synchronized Inference Engine not available in Inference API")
        
        logger.info("✅ All AI components initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI components: {e}")
        logger.error(traceback.format_exc())
        raise


def decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode base64 image to OpenCV format."""
    try:
        # Remove data URL prefix if present
        if image_base64.startswith('data:image'):
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_base64)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format (BGR)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return cv_image
    
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def process_frame_inference(image: np.ndarray, timestamp: float = None) -> InferenceResult:
    """Process a single frame and return inference result with detailed logging."""
    if timestamp is None:
        timestamp = time.time()
    
    inference_start_time = time.time()
    logger.info(f"🔍 Starting inference for frame at timestamp {timestamp}")
    logger.info(f"   📐 Image shape: {image.shape}")
    logger.info(f"   📊 Image dtype: {image.dtype}")
    
    try:
        # Step 1: Face detection
        face_detection_start = time.time()
        logger.info(f"👁️  Step 1: Face detection started")
        
        landmark_data = face_detector.detect_landmarks(image)
        face_detected = landmark_data is not None
        face_detection_time = time.time() - face_detection_start
        
        if not face_detected:
            logger.warning(f"❌ No face detected in frame (detection time: {face_detection_time:.3f}s)")
            features = feature_extractor._get_default_features()
            drowsiness_score = 0.5  # Neutral score
            confidence = 0.0
            alert_level = "no_face"
            
            total_time = time.time() - inference_start_time
            logger.info(f"📊 Final result - Score: {drowsiness_score}, Alert: {alert_level}, Confidence: {confidence}")
            logger.info(f"⏱️  Total time: {total_time:.3f}s")
        else:
            logger.info(f"✅ Face detected successfully (detection time: {face_detection_time:.3f}s)")
            if landmark_data:
                logger.info(f"   📍 Landmarks found: {len(landmark_data.get('all_landmarks', []))}")
                logger.info(f"   👁️  Left eye landmarks: {len(landmark_data.get('left_eye', []))}")
                logger.info(f"   👁️  Right eye landmarks: {len(landmark_data.get('right_eye', []))}")
                logger.info(f"   👄 Mouth landmarks: {len(landmark_data.get('mouth', []))}")
                logger.info(f"   👃 Nose landmarks: {len(landmark_data.get('nose', []))}")
                logger.info(f"   📐 Face outline landmarks: {len(landmark_data.get('face_outline', []))}")
            
            # Step 2: Feature extraction
            feature_extraction_start = time.time()
            logger.info(f"⚙️  Step 2: Feature extraction started")
            
            features = feature_extractor.extract_features(landmark_data, timestamp)
            feature_extraction_time = time.time() - feature_extraction_start
            
            logger.info(f"✅ Features extracted (time: {feature_extraction_time:.3f}s):")
            logger.info(f"   👁️  Left EAR: {features.get('left_ear', 0):.4f}")
            logger.info(f"   👁️  Right EAR: {features.get('right_ear', 0):.4f}")
            logger.info(f"   👁️  Avg EAR: {features.get('avg_ear', 0):.4f} (threshold: 0.25)")
            logger.info(f"   👄 MAR: {features.get('mar', 0):.4f} (threshold: 0.5)")
            logger.info(f"   🎯 Head Pitch: {features.get('head_pitch', 0):.2f}°")
            logger.info(f"   🎯 Head Yaw: {features.get('head_yaw', 0):.2f}°")
            logger.info(f"   🎯 Head Roll: {features.get('head_roll', 0):.2f}°")
            logger.info(f"   👀 Eye Closure: {features.get('eye_closure', 0):.4f}")
            logger.info(f"   🥱 Yawn Indicator: {features.get('yawn_indicator', 0):.4f}")
            logger.info(f"   👁️  Blink Detected: {features.get('blink_detected', 0)}")
            logger.info(f"   🎪 Head Nod Detected: {features.get('nod_detected', 0)}")
            logger.info(f"   👁️  Blink Frequency: {features.get('blink_frequency', 0):.2f}/min")
            logger.info(f"   🎪 Nod Frequency: {features.get('nod_frequency', 0):.2f}/min")
            
            # Step 3: Drowsiness prediction
            prediction_start = time.time()
            logger.info(f"🧠 Step 3: Model prediction started")
            
            if drowsiness_model is not None and feature_scaler is not None:
                logger.info(f"   ✅ Model loaded: {type(drowsiness_model).__name__}")
                logger.info(f"   ✅ Scaler loaded: {type(feature_scaler).__name__}")
                
                # Create feature vector for model (excluding only timestamp)
                feature_vector = []
                feature_names = []
                for key, value in sorted(features.items()):
                    if key not in ['timestamp']:  # Only exclude timestamp, keep face_detected
                        feature_vector.append(float(value))
                        feature_names.append(key)
                
                logger.info(f"📊 Feature vector created:")
                logger.info(f"   🔢 Features count: {len(feature_vector)} (model expects 14)")
                logger.info(f"   📝 Feature names: {feature_names}")
                
                # Expected features from training (for reference)
                expected_features = ['avg_ear', 'blink_detected', 'blink_frequency', 'eye_closure', 
                                   'face_detected', 'head_pitch', 'head_roll', 'head_yaw', 
                                   'left_ear', 'mar', 'nod_detected', 'nod_frequency', 
                                   'right_ear', 'yawn_indicator']
                
                # Check for missing features
                missing_features = set(expected_features) - set(feature_names)
                extra_features = set(feature_names) - set(expected_features)
                
                if missing_features:
                    logger.warning(f"   ⚠️  Missing features: {list(missing_features)}")
                if extra_features:
                    logger.warning(f"   ➕ Extra features: {list(extra_features)}")
                if len(feature_vector) == 14:
                    logger.info(f"   ✅ Feature count matches model expectation!")
                
                # Log feature values in groups for readability
                eye_features = {k: v for k, v in zip(feature_names, feature_vector) if 'ear' in k.lower() or 'blink' in k.lower() or 'eye' in k.lower()}
                mouth_features = {k: v for k, v in zip(feature_names, feature_vector) if 'mar' in k.lower() or 'yawn' in k.lower()}
                head_features = {k: v for k, v in zip(feature_names, feature_vector) if 'head' in k.lower() or 'nod' in k.lower()}
                meta_features = {k: v for k, v in zip(feature_names, feature_vector) if k in ['face_detected']}
                
                if eye_features:
                    logger.info(f"   👁️  Eye features: {', '.join([f'{k}={v:.4f}' for k, v in eye_features.items()])}")
                if mouth_features:
                    logger.info(f"   👄 Mouth features: {', '.join([f'{k}={v:.4f}' for k, v in mouth_features.items()])}")
                if head_features:
                    logger.info(f"   🎯 Head features: {', '.join([f'{k}={v:.4f}' for k, v in head_features.items()])}")
                if meta_features:
                    logger.info(f"   🎭 Meta features: {', '.join([f'{k}={v:.4f}' for k, v in meta_features.items()])}")
                
                # Convert to numpy array and scale features
                X = np.array(feature_vector).reshape(1, -1)
                logger.info(f"   🔄 Raw feature matrix shape: {X.shape}")
                logger.info(f"   📈 Raw feature range: [{X.min():.4f}, {X.max():.4f}]")
                
                try:
                    X_scaled = feature_scaler.transform(X)
                    logger.info(f"   ⚖️  Scaled feature range: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")
                except Exception as scaling_error:
                    logger.error(f"   ❌ Scaling error: {scaling_error}")
                    logger.error(f"   🔍 Feature vector shape: {X.shape}")
                    logger.error(f"   🔍 Scaler expects: {feature_scaler.n_features_in_} features")
                    raise scaling_error
                
                # Get prediction (probability of drowsiness)
                try:
                    probabilities = drowsiness_model.predict_proba(X_scaled)
                    prediction_time = time.time() - prediction_start
                    
                    # FIXED: Swap the interpretation - model outputs [drowsy, alert] not [alert, drowsy]
                    original_drowsiness_score = probabilities[0][1]  # Probability of drowsy class (class 0)
                    alert_probability = probabilities[0][0]  # Probability of alert class (class 1)
                    confidence = max(probabilities[0]) * 2 - 1  # Convert to 0-1 scale
                    
                    logger.info(f"✅ Model prediction completed (time: {prediction_time:.3f}s):")
                    logger.info(f"   🟢 Alert probability: {alert_probability:.4f}")
                    logger.info(f"   🔴 Original drowsy probability: {original_drowsiness_score:.4f}")
                    logger.info(f"   🎯 Confidence: {confidence:.4f}")
                    logger.info(f"   📊 Raw probabilities: {probabilities[0]} [drowsy, alert]")
                    
                    # Use original drowsiness score
                    drowsiness_score = original_drowsiness_score
                    
                except Exception as e:
                    prediction_time = time.time() - prediction_start
                    logger.error(f"❌ Prediction error (time: {prediction_time:.3f}s): {e}")
                    import traceback
                    logger.error(f"   📋 Traceback: {traceback.format_exc()}")
                    drowsiness_score = 0.5
                    confidence = 0.0
            else:
                logger.warning(f"⚠️  Model or scaler not available - using fallback")
                logger.warning(f"   Model loaded: {drowsiness_model is not None}")
                logger.warning(f"   Scaler loaded: {feature_scaler is not None}")
                drowsiness_score = 0.5
                confidence = 0.0
            
            # Determine alert level
            if drowsiness_score < 0.3:
                alert_level = "low"
            elif drowsiness_score < 0.5:
                alert_level = "medium"
            elif drowsiness_score < 0.7:
                alert_level = "high"
            else:
                alert_level = "critical"
                
            logger.info(f"🚨 Alert level determined: {alert_level.upper()}")
            logger.info(f"   📊 Score thresholds: low(<0.3), medium(0.3-0.5), high(0.5-0.7), critical(≥0.7)")
            
            total_inference_time = time.time() - inference_start_time
            logger.info(f"📊 Final result - Score: {drowsiness_score:.4f}, Alert: {alert_level}, Confidence: {confidence:.4f}")
            logger.info(f"⏱️  Total inference time: {total_inference_time:.3f}s")
            logger.info(f"   ⏱️  Breakdown - Face: {face_detection_time:.3f}s, Features: {feature_extraction_time:.3f}s, Prediction: {prediction_time:.3f}s" if 'prediction_time' in locals() else f"   ⏱️  Breakdown - Face: {face_detection_time:.3f}s, Features: {feature_extraction_time:.3f}s, Prediction: N/A")
            
            # Send DV data to synchronized inference engine if available
            if sync_inference_engine is not None:
                print("***************** DV DATA RECEIVED *****************", drowsiness_score)
                try:
                    success = sync_inference_engine.receive_dv_from_inference_api(
                        drowsiness_score=drowsiness_score,
                        timestamp=timestamp,
                        features=features,
                        alert_level=alert_level,
                        confidence=confidence,
                        frame_id=f"inference_api_frame_{timestamp:.3f}"
                    )
                    
                    if success:
                        logger.info(f"📡 DV data sent to sync engine from inference API: score={drowsiness_score:.4f}")
                    else:
                        logger.warning(f"⚠️  Failed to send DV data to sync engine from inference API")
                        
                except Exception as e:
                    logger.error(f"❌ Error sending DV data to sync engine: {e}")
        
        return InferenceResult(
            timestamp=timestamp,
            face_detected=face_detected,
            features=features,
            drowsiness_score=float(drowsiness_score),
            alert_level=alert_level,
            confidence=float(confidence)
        )
    
    except Exception as e:
        total_time = time.time() - inference_start_time
        logger.error(f"❌ Critical error in inference (time: {total_time:.3f}s): {e}")
        logger.error(f"   🏷️  Error type: {type(e).__name__}")
        import traceback
        logger.error(f"   📋 Traceback: {traceback.format_exc()}")
        
        # Return error result
        return InferenceResult(
            timestamp=timestamp,
            face_detected=False,
            features={},
            drowsiness_score=0.5,
            alert_level="error",
            confidence=0.0
        )


@app.post("/inference/single_frame", response_model=InferenceResult)
async def infer_single_frame(frame_data: FrameData):
    """
    Perform inference on a single frame.
    
    Args:
        frame_data: Single frame data with base64 encoded image
        
    Returns:
        Inference result for the frame
    """
    try:
        # Decode image
        image = decode_base64_image(frame_data.image_base64)
        
        # Process frame
        result = process_frame_inference(image, frame_data.timestamp)
        result.frame_id = frame_data.frame_id
        
        return result
        
    except Exception as e:
        logger.error(f"Single frame inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/video_stream", response_model=VideoInferenceResponse)
async def infer_video_stream(video_data: VideoStreamInference):
    """
    Perform inference on a video stream (sequence of frames).
    
    Args:
        video_data: Video stream with frames and processing options
        
    Returns:
        Comprehensive inference results for the entire video
    """
    start_time = time.time()
    
    try:
        frame_results = []
        drowsiness_scores = []
        alert_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        processed_frames = 0
        
        logger.info(f"Processing video stream with {len(video_data.frames)} frames")
        
        for i, frame_data in enumerate(video_data.frames):
            try:
                # Decode image
                image = decode_base64_image(frame_data.image_base64)
                
                # Process frame
                result = process_frame_inference(image, frame_data.timestamp)
                result.frame_id = frame_data.frame_id or f"frame_{i:04d}"
                
                frame_results.append(result)
                drowsiness_scores.append(result.drowsiness_score)
                alert_counts[result.alert_level] = alert_counts.get(result.alert_level, 0) + 1
                processed_frames += 1
                
                # Log progress every 30 frames
                if (i + 1) % 30 == 0:
                    logger.info(f"Processed {i + 1}/{len(video_data.frames)} frames")
                
            except Exception as e:
                logger.error(f"Error processing frame {i}: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        # Calculate summary statistics
        avg_drowsiness = np.mean(drowsiness_scores) if drowsiness_scores else 0.0
        max_drowsiness = np.max(drowsiness_scores) if drowsiness_scores else 0.0
        alert_frames = alert_counts.get("high", 0) + alert_counts.get("critical", 0)
        
        summary = {
            "avg_drowsiness_score": float(avg_drowsiness),
            "max_drowsiness_score": float(max_drowsiness),
            "alert_distribution": alert_counts,
            "face_detection_rate": sum(1 for r in frame_results if r.face_detected) / max(processed_frames, 1),
            "processing_fps": processed_frames / processing_time if processing_time > 0 else 0,
            "recommendations": generate_recommendations(avg_drowsiness, max_drowsiness, alert_frames, processed_frames)
        }
        
        response = VideoInferenceResponse(
            total_frames=len(video_data.frames),
            processed_frames=processed_frames,
            processing_time=processing_time,
            avg_drowsiness_score=float(avg_drowsiness),
            max_drowsiness_score=float(max_drowsiness),
            alert_frames=alert_frames,
            frame_results=frame_results,
            summary=summary
        )
        
        logger.info(f"Video inference completed: {processed_frames} frames, "
                   f"avg_score={avg_drowsiness:.3f}, time={processing_time:.2f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"Video stream inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/upload_video")
async def infer_uploaded_video(
    video_file: UploadFile = File(...),
    extract_fps: float = Form(5.0),
    max_frames: int = Form(300)
):
    """
    Perform inference on an uploaded video file.
    
    Args:
        video_file: Uploaded video file
        extract_fps: Frame extraction rate
        max_frames: Maximum frames to process
        
    Returns:
        Video inference results
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await video_file.read()
            temp_file.write(content)
            temp_video_path = temp_file.name
        
        try:
            # Extract frames from video
            frames = extract_frames_from_video(temp_video_path, extract_fps, max_frames)
            
            # Convert to FrameData format
            frame_data_list = []
            for i, frame in enumerate(frames):
                # Encode frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                frame_data_list.append(FrameData(
                    image_base64=frame_base64,
                    timestamp=i / extract_fps,
                    frame_id=f"frame_{i:04d}"
                ))
            
            # Create video stream inference request
            video_inference = VideoStreamInference(
                frames=frame_data_list,
                processing_options={
                    "source": "uploaded_video",
                    "original_filename": video_file.filename,
                    "extract_fps": extract_fps,
                    "max_frames": max_frames
                }
            )
            
            # Process video
            result = await infer_video_stream(video_inference)
            
            return result
            
        finally:
            # Clean up temporary file
            os.unlink(temp_video_path)
        
    except Exception as e:
        logger.error(f"Video upload inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def extract_frames_from_video(video_path: str, fps: float, max_frames: int) -> List[np.ndarray]:
    """Extract frames from video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Calculate frame interval
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(original_fps / fps))
    
    frame_count = 0
    extracted_count = 0
    
    while cap.read()[0] and extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append(frame)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {len(frames)} frames from video at {fps} FPS")
    
    return frames


def generate_recommendations(avg_score: float, max_score: float, alert_frames: int, total_frames: int) -> List[str]:
    """Generate recommendations based on inference results."""
    recommendations = []
    
    if avg_score > 0.7:
        recommendations.append("HIGH RISK: Significant drowsiness detected. Immediate rest recommended.")
    elif avg_score > 0.5:
        recommendations.append("MODERATE RISK: Some drowsiness detected. Consider taking a break.")
    elif avg_score > 0.3:
        recommendations.append("LOW RISK: Mild fatigue signs detected. Monitor alertness.")
    else:
        recommendations.append("ALERT: No significant drowsiness detected.")
    
    if max_score > 0.8:
        recommendations.append("Critical drowsiness peak detected during session.")
    
    alert_rate = alert_frames / max(total_frames, 1)
    if alert_rate > 0.2:
        recommendations.append(f"High alert frequency: {alert_rate:.1%} of frames showed drowsiness.")
    
    if total_frames < 25:  # Less than 5 seconds at 5fps
        recommendations.append("Short video duration. Consider longer monitoring for accurate assessment.")
    
    return recommendations


@app.post("/inference/batch", response_model=List[VideoInferenceResponse])
async def infer_batch_videos(batch_request: BatchInferenceRequest):
    """
    Perform batch inference on multiple video streams.
    
    Args:
        batch_request: Batch of video streams for processing
        
    Returns:
        List of video inference results
    """
    try:
        results = []
        batch_start = time.time()
        
        logger.info(f"Processing batch with {len(batch_request.videos)} videos")
        
        for i, video_data in enumerate(batch_request.videos):
            logger.info(f"Processing batch video {i+1}/{len(batch_request.videos)}")
            result = await infer_video_stream(video_data)
            results.append(result)
        
        batch_time = time.time() - batch_start
        logger.info(f"Batch processing completed in {batch_time:.2f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inference/health")
async def health_check():
    """Health check for inference service."""
    return {
        "status": "healthy",
        "service": "Vigilant AI Inference API",
        "version": "3.0.0",
        "ai_components": {
            "face_detector": face_detector is not None,
            "feature_extractor": feature_extractor is not None,
            "window_processor": window_processor is not None,
            "drowsiness_model": drowsiness_model is not None,
            "sync_inference_engine": sync_inference_engine is not None
        },
        "model_info": {
            "model_loaded": drowsiness_model is not None,
            "scaler_loaded": feature_scaler is not None,
            "model_type": type(drowsiness_model).__name__ if drowsiness_model else "none"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/inference/sync_engine_status")
async def get_inference_sync_engine_status():
    """Get status of the synchronized inference engine in inference API."""
    if sync_inference_engine is None:
        return {
            "status": "not_available",
            "message": "Synchronized Inference Engine not initialized in Inference API"
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
            "service": "inference_api",
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
        logger.error(f"Error getting sync engine status in inference API: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Vigilant AI - Inference API",
        "version": "3.0.0",
        "description": "Drowsiness detection inference from video streams",
        "endpoints": {
            "single_frame": "/inference/single_frame",
            "video_stream": "/inference/video_stream", 
            "upload_video": "/inference/upload_video",
            "batch": "/inference/batch",
            "sync_engine_status": "/inference/sync_engine_status",
            "health": "/inference/health",
            "docs": "/docs"
        },
        "usage": {
            "single_frame": "POST base64 encoded image for single frame inference",
            "video_stream": "POST sequence of frames for video analysis",
            "upload_video": "POST video file for automated frame extraction and analysis",
            "batch": "POST multiple video streams for batch processing"
        }
    }


if __name__ == "__main__":
    # Run the inference server
    uvicorn.run(
        "inference_api:app",
        host="0.0.0.0",
        port=8002,  # Different port from other services
        reload=True,
        log_level="info"
    )


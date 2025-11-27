"""
Streaming API Routes
====================
WebSocket endpoints for real-time camera streaming and detection.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
import asyncio
import uuid
import time
import cv2
import base64
import tempfile
from pathlib import Path

from ..core.pipeline import FoodDetectionPipeline
from ..streaming.camera import CameraCapture
from ..streaming.processor import FrameProcessor
from ..streaming.websocket import WebSocketManager


# Create router
router = APIRouter(prefix="/api/v1/stream", tags=["streaming"])

# Global instances
_pipeline = None
_camera = None
_processor = None
_ws_manager = WebSocketManager()
_streaming_task = None


def get_pipeline():
    """Get or initialize pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = FoodDetectionPipeline()
    return _pipeline


async def start_streaming(camera_id: int = 0, skip_frames: int = 15, conf: float = 0.25):
    """
    Start camera streaming and processing.
    
    Args:
        camera_id: Camera device ID
        skip_frames: Process every Nth frame (15 = ~2 FPS detection)
                    Recommended: 10-20 for lag-free streaming
        conf: Detection confidence threshold
    """
    global _camera, _processor, _streaming_task
    
    if _camera and _camera.is_running:
        print("[Streaming] Already running")
        # Update parameters if running
        if _processor:
            _processor.skip_frames = skip_frames
            _processor.conf = conf
        return
    
    # Initialize camera
    _camera = CameraCapture(camera_id=camera_id, width=640, height=480, fps=30)
    if not _camera.start():
        raise RuntimeError(f"Failed to open camera {camera_id}")
    
    # Initialize processor with optimized settings for lag-free streaming
    pipeline = get_pipeline()
    _processor = FrameProcessor(
        pipeline, 
        skip_frames=skip_frames,
        max_queue_size=1,              # Keep queue size ≤ 1 (no backlog)
        conf=conf,
        enable_change_detection=True,  # Enable SSIM-based pre-filter
        auto_flush_queue=True          # Auto-flush old frames
    )
    _processor.start()
    
    # Start streaming loop
    _streaming_task = asyncio.create_task(_streaming_loop())
    
    print(f"[Streaming] Started (camera_id={camera_id}, skip_frames={skip_frames}, conf={conf})")


async def stop_streaming():
    """Stop camera streaming and processing."""
    global _camera, _processor, _streaming_task
    
    # Cancel streaming task
    if _streaming_task:
        _streaming_task.cancel()
        try:
            await _streaming_task
        except asyncio.CancelledError:
            pass
        _streaming_task = None
    
    # Stop processor
    if _processor:
        _processor.stop()
        _processor = None
    
    # Stop camera
    if _camera:
        _camera.stop()
        _camera = None
    
    print("[Streaming] Stopped")


async def _streaming_loop():
    """
    Dual-mode streaming loop:
    1. Send preview frames continuously (low quality, ~10fps) for smooth video
    2. Capture + detect every N seconds and send annotated results
    """
    last_stats_time = time.time()
    last_detection_time = 0
    last_preview_time = 0
    
    stats_interval = 10.0  # Broadcast stats every 10 seconds
    detection_interval = _processor.skip_frames / 30.0  # Convert frames to seconds
    preview_interval = 0.1  # Send preview every 100ms (~10fps)
    
    try:
        while True:
            current_time = time.time()
            frame = _camera.get_frame()
            
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            
            # 1. Send preview frames (low quality, frequent)
            if current_time - last_preview_time >= preview_interval:
                try:
                    # Resize and compress for preview
                    preview_frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
                    _, buffer = cv2.imencode('.jpg', preview_frame, encode_param)
                    preview_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send as preview type
                    await _ws_manager.broadcast({
                        'type': 'preview',
                        'data': preview_base64,
                        'timestamp': current_time
                    })
                    
                    last_preview_time = current_time
                except Exception as e:
                    print(f"[Preview] Error: {e}")
            
            # 2. Capture + Detect at interval (high quality)
            if current_time - last_detection_time >= detection_interval:
                try:
                    # Notify user
                    await _ws_manager.broadcast_text("📸 Capturing and detecting...")
                    
                    # Save frame temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        cv2.imwrite(tmp.name, frame)
                        tmp_path = tmp.name
                    
                    # Run detection
                    start_time = time.time()
                    results = _processor.pipeline.process_image(tmp_path, conf=_processor.conf)
                    processing_time = time.time() - start_time
                    
                    # Clean up
                    Path(tmp_path).unlink(missing_ok=True)
                    
                    # Annotate frame with detections
                    annotated_frame = _processor._annotate_frame(frame.copy(), results['detections'])
                    
                    # Encode to base64 (higher quality)
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                    _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
                    result_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Broadcast detection result
                    await _ws_manager.broadcast({
                        'type': 'detection',
                        'data': result_base64,
                        'detections': results['detections'],
                        'count': len(results['detections']),
                        'processing_time': round(processing_time, 3),
                        'timestamp': current_time
                    })
                    
                    _processor.frames_processed += 1
                    print(f"[Detection] Found {len(results['detections'])} items in {processing_time:.2f}s")
                    
                    last_detection_time = current_time
                    
                except Exception as e:
                    print(f"[Detection] Error: {e}")
                    await _ws_manager.broadcast_error(f"Detection error: {str(e)}")
            
            # 3. Broadcast stats periodically
            if current_time - last_stats_time >= stats_interval:
                camera_info = _camera.get_frame_info()
                processor_stats = _processor.get_stats()
                await _ws_manager.broadcast_stats(camera_info, processor_stats)
                last_stats_time = current_time
            
            # Small delay
            await asyncio.sleep(0.01)
    
    except asyncio.CancelledError:
        print("[Streaming] Loop cancelled")
        raise


@router.websocket("/camera")
async def websocket_camera_stream(
    websocket: WebSocket,
    camera_id: int = Query(0, description="Camera device ID"),
    skip_frames: int = Query(150, ge=1, le=500, description="Process every Nth frame (30fps: 150=5s, 300=10s)"),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Detection confidence threshold")
):
    """
    WebSocket endpoint for camera streaming.
    
    Query Parameters:
        camera_id: Camera device ID (default: 0)
        skip_frames: Process every Nth frame (1-500, default: 150 = 5 seconds at 30fps)
                     Examples: 30=1s, 60=2s, 150=5s, 300=10s
        conf: Detection confidence threshold (0.0-1.0, default: 0.25)
    
    Message Format (sent to client):
        {
            "type": "image" | "json" | "text" | "error" | "stats",
            "data": <base64_image> | <json_object> | <string>,
            "detections": [...],  // only for type="image"
            "count": <int>,       // only for type="image"
            "processing_time": <float>,  // only for type="image"
            "timestamp": <float>
        }
    """
    # Generate unique client ID
    client_id = str(uuid.uuid4())[:8]
    
    # Connect client
    await _ws_manager.connect(websocket, client_id)
    
    try:
        # Start streaming if not already running
        if not _camera or not _camera.is_running:
            await start_streaming(camera_id=camera_id, skip_frames=skip_frames, conf=conf)
        else:
            # Update parameters if already running
            if _processor:
                _processor.skip_frames = skip_frames
                _processor.conf = conf
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive messages from client (e.g., control commands)
                data = await websocket.receive_text()
                
                # Parse command
                import json
                try:
                    command = json.loads(data)
                    await _handle_client_command(websocket, command)
                except json.JSONDecodeError:
                    await _ws_manager.send_message(websocket, {
                        'type': 'error',
                        'data': 'Invalid JSON command',
                        'timestamp': time.time()
                    })
            
            except WebSocketDisconnect:
                break
    
    except Exception as e:
        print(f"[WebSocket] Error in camera stream: {e}")
    
    finally:
        # Disconnect client
        _ws_manager.disconnect(websocket)
        
        # Stop streaming if no more clients
        if _ws_manager.get_connection_count() == 0:
            await stop_streaming()


async def _handle_client_command(websocket: WebSocket, command: dict):
    """
    Handle control commands from client.
    
    Args:
        websocket: Client WebSocket
        command: Command dict with 'action' key
    """
    action = command.get('action')
    
    if action == 'get_stats':
        # Send current stats
        if _camera and _processor:
            camera_info = _camera.get_frame_info()
            processor_stats = _processor.get_stats()
            ws_stats = _ws_manager.get_stats()
            
            await _ws_manager.send_message(websocket, {
                'type': 'stats',
                'data': {
                    'camera': camera_info,
                    'processor': processor_stats,
                    'websocket': ws_stats
                },
                'timestamp': time.time()
            })
    
    elif action == 'pause':
        # Pause processing (future feature)
        await _ws_manager.send_message(websocket, {
            'type': 'text',
            'data': 'Pause not implemented yet',
            'timestamp': time.time()
        })
    
    else:
        await _ws_manager.send_message(websocket, {
            'type': 'error',
            'data': f'Unknown action: {action}',
            'timestamp': time.time()
        })


@router.get("/cameras")
async def list_cameras():
    """
    List available camera devices.
    
    Returns:
        JSON with list of available camera IDs
    """
    try:
        cameras = CameraCapture.list_cameras()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "cameras": cameras,
                "count": len(cameras)
            }
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@router.get("/status")
async def streaming_status():
    """
    Get current streaming status.
    
    Returns:
        JSON with streaming status and stats
    """
    is_running = _camera is not None and _camera.is_running
    
    data = {
        "streaming": is_running,
        "connections": _ws_manager.get_connection_count()
    }
    
    if is_running:
        data["camera"] = _camera.get_frame_info()
        data["processor"] = _processor.get_stats()
        data["websocket"] = _ws_manager.get_stats()
    
    return JSONResponse(content={
        "success": True,
        "data": data
    })


@router.post("/stop")
async def stop_streaming_endpoint():
    """
    Manually stop streaming.
    
    Returns:
        JSON with success status
    """
    try:
        await stop_streaming()
        
        return JSONResponse(content={
            "success": True,
            "message": "Streaming stopped"
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

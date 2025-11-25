"""
Training Data Collection API
=============================
WebSocket endpoint for streaming raw images for dataset building.
Used for collecting training data and future labeling.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, UploadFile, File, Form
from fastapi.responses import JSONResponse
import asyncio
import uuid
import time
import cv2
import base64
from pathlib import Path
from typing import Optional
import json

from ..streaming.camera import CameraCapture
from ..streaming.websocket import WebSocketManager


router = APIRouter(prefix="/api/v1/training", tags=["training"])

# Global instances
_camera = None
_ws_manager = WebSocketManager()
_streaming_task = None

# Training data storage
TRAINING_DATA_DIR = Path("data/training_images")
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)


async def start_training_stream(camera_id: int = 0, interval_seconds: float = 5.0):
    """
    Start camera streaming for training data collection.
    
    Args:
        camera_id: Camera device ID
        interval_seconds: Capture interval in seconds
    """
    global _camera, _streaming_task
    
    if _camera and _camera.is_running:
        print("[Training] Already running")
        return
    
    # Initialize camera
    _camera = CameraCapture(camera_id=camera_id, width=640, height=480, fps=30)
    if not _camera.start():
        raise RuntimeError(f"Failed to open camera {camera_id}")
    
    # Start streaming loop
    _streaming_task = asyncio.create_task(_training_loop(interval_seconds))
    
    print(f"[Training] Started (camera_id={camera_id}, interval={interval_seconds}s)")


async def stop_training_stream():
    """Stop training streaming."""
    global _camera, _streaming_task
    
    # Cancel streaming task
    if _streaming_task:
        _streaming_task.cancel()
        try:
            await _streaming_task
        except asyncio.CancelledError:
            pass
        _streaming_task = None
    
    # Stop camera
    if _camera:
        _camera.stop()
        _camera = None
    
    print("[Training] Stopped")


async def _training_loop(interval_seconds: float):
    """
    Main training loop: capture frame → encode → send image.
    """
    last_capture_time = 0
    capture_count = 0
    
    try:
        while True:
            current_time = time.time()
            
            # Check if it's time to capture
            if current_time - last_capture_time >= interval_seconds:
                frame = _camera.get_frame()
                
                if frame is not None:
                    try:
                        capture_count += 1
                        
                        # Encode frame to JPEG
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                        _, buffer = cv2.imencode('.jpg', frame, encode_param)
                        image_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Generate image ID
                        image_id = f"train_{int(current_time)}_{capture_count:04d}"
                        
                        # Broadcast image
                        message = {
                            'type': 'training_image',
                            'image_id': image_id,
                            'camera_id': f'cam-{_camera.camera_id:02d}',
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'image': image_base64,
                            'capture_count': capture_count
                        }
                        
                        await _ws_manager.broadcast(message)
                        
                        print(f"[Training] Captured image #{capture_count} (id: {image_id})")
                        
                        last_capture_time = current_time
                        
                    except Exception as e:
                        print(f"[Training] Error: {e}")
                        await _ws_manager.broadcast_error(f"Capture error: {str(e)}")
            
            # Small delay
            await asyncio.sleep(0.1)
    
    except asyncio.CancelledError:
        print("[Training] Loop cancelled")
        raise


@router.websocket("/stream")
async def websocket_training_stream(
    websocket: WebSocket,
    camera_id: int = Query(0, description="Camera device ID"),
    interval: float = Query(5.0, ge=1.0, le=30.0, description="Capture interval in seconds")
):
    """
    WebSocket endpoint for training data streaming.
    Streams raw images for dataset collection.
    
    Query Parameters:
        camera_id: Camera device ID (default: 0)
        interval: Capture interval in seconds (1.0-30.0, default: 5.0)
    
    Message Format (sent to client):
        {
            "type": "training_image",
            "image_id": "train_1732531845_0001",
            "camera_id": "cam-01",
            "timestamp": "2025-11-25 15:30:45",
            "image": "<base64_encoded_jpeg>",
            "capture_count": 1
        }
    """
    # Generate unique client ID
    client_id = str(uuid.uuid4())[:8]
    
    # Connect client
    await _ws_manager.connect(websocket, client_id)
    
    try:
        # Start streaming if not already running
        if not _camera or not _camera.is_running:
            await start_training_stream(camera_id=camera_id, interval_seconds=interval)
        
        # Keep connection alive
        while True:
            try:
                # Receive messages from client
                data = await websocket.receive_text()
                
                # Parse command (for future use)
                try:
                    command = json.loads(data)
                    if command.get('action') == 'get_stats':
                        await _ws_manager.send_message(websocket, {
                            'type': 'stats',
                            'data': {
                                'connections': _ws_manager.get_connection_count(),
                                'camera_running': _camera.is_running if _camera else False
                            },
                            'timestamp': time.time()
                        })
                except json.JSONDecodeError:
                    pass
            
            except WebSocketDisconnect:
                break
    
    except Exception as e:
        print(f"[Training] WebSocket error: {e}")
    
    finally:
        # Disconnect client
        _ws_manager.disconnect(websocket)
        
        # Stop streaming if no more clients
        if _ws_manager.get_connection_count() == 0:
            await stop_training_stream()


@router.post("/upload")
async def upload_training_image(
    image: UploadFile = File(...),
    label: Optional[str] = Form(None),
    image_id: Optional[str] = Form(None)
):
    """
    Upload a training image with optional label.
    
    This endpoint is prepared for future labeling workflow:
    - Client captures images via /training/stream
    - User labels images in UI
    - Client uploads labeled images back to server
    
    Args:
        image: Image file
        label: Food class label (optional, for future use)
        image_id: Original image ID from stream (optional)
    
    Returns:
        JSON with upload status and saved file path
    """
    try:
        # Generate filename
        timestamp = int(time.time())
        filename = f"train_{timestamp}_{image.filename}"
        
        if label:
            # Save to class-specific directory
            save_dir = TRAINING_DATA_DIR / label
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Save to unlabeled directory
            save_dir = TRAINING_DATA_DIR / "unlabeled"
            save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / filename
        
        # Save file
        with open(save_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        print(f"[Training] Uploaded: {save_path} (label: {label or 'unlabeled'})")
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "filename": filename,
                "path": str(save_path),
                "label": label,
                "image_id": image_id,
                "size": len(content)
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
async def training_status():
    """
    Get current training streaming status.
    
    Returns:
        JSON with streaming status and statistics
    """
    is_running = _camera is not None and _camera.is_running
    
    # Count training images
    total_images = 0
    images_by_class = {}
    
    for class_dir in TRAINING_DATA_DIR.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg")))
            images_by_class[class_dir.name] = count
            total_images += count
    
    data = {
        "streaming": is_running,
        "connections": _ws_manager.get_connection_count(),
        "storage": {
            "total_images": total_images,
            "images_by_class": images_by_class,
            "storage_path": str(TRAINING_DATA_DIR)
        }
    }
    
    if is_running:
        data["camera"] = _camera.get_frame_info()
    
    return JSONResponse(content={
        "success": True,
        "data": data
    })


@router.post("/stop")
async def stop_training_endpoint():
    """
    Manually stop training streaming.
    
    Returns:
        JSON with success status
    """
    try:
        await stop_training_stream()
        
        return JSONResponse(content={
            "success": True,
            "message": "Training streaming stopped"
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@router.get("/images")
async def list_training_images(
    label: Optional[str] = Query(None, description="Filter by label")
):
    """
    List all training images.
    
    Args:
        label: Filter by specific label (optional)
    
    Returns:
        JSON with list of training images
    """
    try:
        images = []
        
        if label:
            # List images for specific label
            label_dir = TRAINING_DATA_DIR / label
            if label_dir.exists():
                for img_path in label_dir.glob("*.jpg"):
                    images.append({
                        "filename": img_path.name,
                        "path": str(img_path),
                        "label": label,
                        "size": img_path.stat().st_size
                    })
        else:
            # List all images
            for class_dir in TRAINING_DATA_DIR.iterdir():
                if class_dir.is_dir():
                    for img_path in class_dir.glob("*.jpg"):
                        images.append({
                            "filename": img_path.name,
                            "path": str(img_path),
                            "label": class_dir.name,
                            "size": img_path.stat().st_size
                        })
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "images": images,
                "count": len(images)
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

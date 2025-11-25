"""
Prediction Streaming API
========================
WebSocket endpoint for real-time food detection results (JSON only, no images).
Used for production billing/POS systems.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
import asyncio
import uuid
import time
import cv2
import tempfile
from pathlib import Path
from typing import Dict, List
import json

from ..core.pipeline import FoodDetectionPipeline
from ..streaming.camera import CameraCapture
from ..streaming.websocket import WebSocketManager


router = APIRouter(prefix="/api/v1/predict", tags=["prediction"])

# Global instances
_pipeline = None
_camera = None
_ws_manager = WebSocketManager()
_streaming_task = None

# Price database (TODO: move to database)
FOOD_PRICES = {
    'coconut': 20000,
    'cua': 25000,
    'macaron': 15000,
    'meden': 18000,
    'melon': 30000,
}


def get_pipeline():
    """Get or initialize pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = FoodDetectionPipeline()
    return _pipeline


async def start_prediction_stream(camera_id: int = 0, interval_seconds: float = 2.0, conf: float = 0.25):
    """
    Start camera streaming for prediction results.
    
    Args:
        camera_id: Camera device ID
        interval_seconds: Detection interval in seconds
        conf: Detection confidence threshold
    """
    global _camera, _streaming_task
    
    if _camera and _camera.is_running:
        print("[Predict] Already running")
        return
    
    # Initialize camera
    _camera = CameraCapture(camera_id=camera_id, width=640, height=480, fps=30)
    if not _camera.start():
        raise RuntimeError(f"Failed to open camera {camera_id}")
    
    # Start streaming loop
    _streaming_task = asyncio.create_task(_prediction_loop(interval_seconds, conf))
    
    print(f"[Predict] Started (camera_id={camera_id}, interval={interval_seconds}s, conf={conf})")


async def stop_prediction_stream():
    """Stop prediction streaming."""
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
    
    print("[Predict] Stopped")


async def _prediction_loop(interval_seconds: float, conf: float):
    """
    Main prediction loop: capture frame → detect → send JSON result.
    """
    last_detection_time = 0
    pipeline = get_pipeline()
    
    try:
        while True:
            current_time = time.time()
            
            # Check if it's time to detect
            if current_time - last_detection_time >= interval_seconds:
                frame = _camera.get_frame()
                
                if frame is not None:
                    try:
                        # Notify clients
                        await _ws_manager.broadcast_text("📸 Detecting...")
                        
                        # Save frame temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                            cv2.imwrite(tmp.name, frame)
                            tmp_path = tmp.name
                        
                        # Run detection
                        start_time = time.time()
                        results = pipeline.process_image(tmp_path, conf=conf, save_to_db=False)
                        processing_time = time.time() - start_time
                        
                        # Clean up
                        Path(tmp_path).unlink(missing_ok=True)
                        
                        # Build result JSON
                        items = []
                        total_price = 0
                        
                        # Count items by class
                        from collections import Counter
                        class_counts = Counter([det['class'] for det in results['detections']])
                        
                        for class_name, qty in class_counts.items():
                            price = FOOD_PRICES.get(class_name, 0)
                            items.append({
                                'name': class_name,
                                'qty': qty,
                                'price': price
                            })
                            total_price += price * qty
                        
                        # Broadcast result (JSON only, no image)
                        result_message = {
                            'type': 'prediction',
                            'camera_id': f'cam-{_camera.camera_id:02d}',
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'items': items,
                            'total_price': total_price,
                            'processing_time': round(processing_time, 3)
                        }
                        
                        await _ws_manager.broadcast(result_message)
                        
                        print(f"[Predict] Detected {len(results['detections'])} items, Total: {total_price}đ")
                        
                        last_detection_time = current_time
                        
                    except Exception as e:
                        print(f"[Predict] Error: {e}")
                        await _ws_manager.broadcast_error(f"Detection error: {str(e)}")
            
            # Small delay
            await asyncio.sleep(0.1)
    
    except asyncio.CancelledError:
        print("[Predict] Loop cancelled")
        raise


@router.websocket("/stream")
async def websocket_prediction_stream(
    websocket: WebSocket,
    camera_id: int = Query(0, description="Camera device ID"),
    interval: float = Query(2.0, ge=0.5, le=10.0, description="Detection interval in seconds"),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Detection confidence threshold")
):
    """
    WebSocket endpoint for prediction streaming.
    Streams JSON results only (no images).
    
    Query Parameters:
        camera_id: Camera device ID (default: 0)
        interval: Detection interval in seconds (0.5-10.0, default: 2.0)
        conf: Detection confidence threshold (0.0-1.0, default: 0.25)
    
    Message Format (sent to client):
        {
            "type": "prediction",
            "camera_id": "cam-01",
            "timestamp": "2025-11-25 15:30:45",
            "items": [
                {"name": "coconut", "qty": 1, "price": 20000},
                {"name": "melon", "qty": 1, "price": 30000}
            ],
            "total_price": 50000,
            "processing_time": 2.345
        }
    """
    # Generate unique client ID
    client_id = str(uuid.uuid4())[:8]
    
    # Connect client
    await _ws_manager.connect(websocket, client_id)
    
    try:
        # Start streaming if not already running
        if not _camera or not _camera.is_running:
            await start_prediction_stream(camera_id=camera_id, interval_seconds=interval, conf=conf)
        
        # Keep connection alive
        while True:
            try:
                # Receive messages from client (optional commands)
                data = await websocket.receive_text()
                
                # Parse command
                try:
                    command = json.loads(data)
                    if command.get('action') == 'get_prices':
                        await _ws_manager.send_message(websocket, {
                            'type': 'prices',
                            'data': FOOD_PRICES,
                            'timestamp': time.time()
                        })
                except json.JSONDecodeError:
                    pass
            
            except WebSocketDisconnect:
                break
    
    except Exception as e:
        print(f"[Predict] WebSocket error: {e}")
    
    finally:
        # Disconnect client
        _ws_manager.disconnect(websocket)
        
        # Stop streaming if no more clients
        if _ws_manager.get_connection_count() == 0:
            await stop_prediction_stream()


@router.get("/status")
async def prediction_status():
    """
    Get current prediction streaming status.
    
    Returns:
        JSON with streaming status
    """
    is_running = _camera is not None and _camera.is_running
    
    data = {
        "streaming": is_running,
        "connections": _ws_manager.get_connection_count()
    }
    
    if is_running:
        data["camera"] = _camera.get_frame_info()
    
    return JSONResponse(content={
        "success": True,
        "data": data
    })


@router.post("/stop")
async def stop_prediction_endpoint():
    """
    Manually stop prediction streaming.
    
    Returns:
        JSON with success status
    """
    try:
        await stop_prediction_stream()
        
        return JSONResponse(content={
            "success": True,
            "message": "Prediction streaming stopped"
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@router.get("/prices")
async def get_food_prices():
    """
    Get food price database.
    
    Returns:
        JSON with food prices
    """
    return JSONResponse(content={
        "success": True,
        "data": FOOD_PRICES
    })

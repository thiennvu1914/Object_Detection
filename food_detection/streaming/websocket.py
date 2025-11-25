"""
WebSocket Manager Module
========================
WebSocket message handling and broadcasting.
"""
from typing import Dict, Any, List, Optional
from fastapi import WebSocket
import json
import time
import asyncio


class WebSocketManager:
    """
    WebSocket connection manager.
    
    Features:
    - Multiple client connections
    - Message type routing (image/json/text)
    - Broadcast to all clients
    - Connection lifecycle management
    """
    
    def __init__(self):
        """Initialize WebSocket manager."""
        self.active_connections: List[WebSocket] = []
        self.connection_ids: Dict[WebSocket, str] = {}
        self.message_count = 0
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """
        Accept and register new WebSocket connection.
        
        Args:
            websocket: FastAPI WebSocket instance
            client_id: Unique client identifier
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_ids[websocket] = client_id
        
        print(f"[WebSocket] Client connected: {client_id} (total: {len(self.active_connections)})")
        
        # Send welcome message
        await self.send_message(websocket, {
            'type': 'text',
            'data': f'Connected to camera stream (ID: {client_id})',
            'timestamp': time.time()
        })
    
    def disconnect(self, websocket: WebSocket):
        """
        Remove WebSocket connection.
        
        Args:
            websocket: FastAPI WebSocket instance
        """
        if websocket in self.active_connections:
            client_id = self.connection_ids.get(websocket, 'unknown')
            self.active_connections.remove(websocket)
            self.connection_ids.pop(websocket, None)
            print(f"[WebSocket] Client disconnected: {client_id} (remaining: {len(self.active_connections)})")
    
    async def send_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        Send message to specific WebSocket client.
        
        Args:
            websocket: Target WebSocket
            message: Message dict with 'type' and 'data' keys
        """
        try:
            await websocket.send_json(message)
            self.message_count += 1
        except Exception as e:
            print(f"[WebSocket] Error sending message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast message to all connected clients.
        
        Args:
            message: Message dict to broadcast
        """
        if not self.active_connections:
            return
        
        # Send to all clients concurrently
        disconnected = []
        
        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"[WebSocket] Error broadcasting to {self.connection_ids.get(websocket)}: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self.disconnect(websocket)
        
        if self.active_connections:
            self.message_count += 1
    
    async def broadcast_image(self, base64_image: str, detections: list, processing_time: float):
        """
        Broadcast detection result with annotated image.
        
        Args:
            base64_image: Base64 encoded JPEG image
            detections: List of detection dicts
            processing_time: Processing time in seconds
        """
        message = {
            'type': 'image',
            'data': base64_image,
            'detections': detections,
            'count': len(detections),
            'processing_time': processing_time,
            'timestamp': time.time()
        }
        
        await self.broadcast(message)
    
    async def broadcast_json(self, data: Dict[str, Any]):
        """
        Broadcast JSON data only (no image).
        
        Args:
            data: JSON-serializable dict
        """
        message = {
            'type': 'json',
            'data': data,
            'timestamp': time.time()
        }
        
        await self.broadcast(message)
    
    async def broadcast_text(self, text: str):
        """
        Broadcast text message.
        
        Args:
            text: Text message string
        """
        message = {
            'type': 'text',
            'data': text,
            'timestamp': time.time()
        }
        
        await self.broadcast(message)
    
    async def broadcast_error(self, error: str):
        """
        Broadcast error message.
        
        Args:
            error: Error message string
        """
        message = {
            'type': 'error',
            'data': error,
            'timestamp': time.time()
        }
        
        await self.broadcast(message)
    
    async def broadcast_stats(self, camera_info: dict, processor_stats: dict):
        """
        Broadcast system statistics.
        
        Args:
            camera_info: Camera status dict
            processor_stats: Processor stats dict
        """
        message = {
            'type': 'stats',
            'data': {
                'camera': camera_info,
                'processor': processor_stats,
                'websocket': {
                    'active_connections': len(self.active_connections),
                    'messages_sent': self.message_count
                }
            },
            'timestamp': time.time()
        }
        
        await self.broadcast(message)
    
    def get_connection_count(self) -> int:
        """
        Get number of active connections.
        
        Returns:
            Number of connected clients
        """
        return len(self.active_connections)
    
    def get_stats(self) -> dict:
        """
        Get WebSocket manager statistics.
        
        Returns:
            Dict with connection stats
        """
        return {
            'active_connections': len(self.active_connections),
            'messages_sent': self.message_count,
            'client_ids': list(self.connection_ids.values())
        }

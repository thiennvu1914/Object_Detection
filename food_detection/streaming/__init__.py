"""
Streaming Module
================
Real-time camera streaming with WebSocket support.
"""
from .camera import CameraCapture
from .processor import FrameProcessor
from .websocket import WebSocketManager

__all__ = ['CameraCapture', 'FrameProcessor', 'WebSocketManager']

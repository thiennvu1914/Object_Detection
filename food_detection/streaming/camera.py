"""
Camera Capture Module
=====================
OpenCV-based camera capture for real-time streaming.
"""
import cv2
import numpy as np
from typing import Optional, Tuple
import threading
import time


class CameraCapture:
    """
    Camera capture handler with OpenCV.
    
    Features:
    - Auto-detect available cameras
    - Thread-safe frame access
    - FPS control
    - Resolution configuration
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):
        """
        Initialize camera capture.
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frames per second
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = fps
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.capture_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.frame_count = 0
        self.start_time = None
        
    def start(self) -> bool:
        """
        Start camera capture.
        
        Returns:
            True if camera opened successfully, False otherwise
        """
        if self.is_running:
            print(f"[Camera] Already running")
            return True
        
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"[Camera] Failed to open camera {self.camera_id}")
            return False
        
        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Get actual settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"[Camera] Opened camera {self.camera_id}")
        print(f"[Camera] Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
        
        # Start capture thread
        self.is_running = True
        self.start_time = time.time()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        return True
    
    def stop(self):
        """Stop camera capture and release resources."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for thread to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Calculate statistics
        if self.start_time:
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"[Camera] Stopped. Captured {self.frame_count} frames in {elapsed:.1f}s (avg {avg_fps:.1f} FPS)")
    
    def _capture_loop(self):
        """Internal capture loop (runs in separate thread)."""
        frame_interval = 1.0 / self.target_fps
        
        while self.is_running:
            loop_start = time.time()
            
            ret, frame = self.cap.read()
            
            if not ret:
                print(f"[Camera] Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Update current frame (thread-safe)
            with self.frame_lock:
                self.current_frame = frame.copy()
                self.frame_count += 1
            
            # FPS throttling
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get latest captured frame.
        
        Returns:
            Frame as numpy array (BGR format), or None if not available
        """
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
    
    def get_frame_info(self) -> dict:
        """
        Get camera and frame information.
        
        Returns:
            Dict with camera status, resolution, FPS, etc.
        """
        if not self.is_running or not self.cap:
            return {
                "status": "stopped",
                "camera_id": self.camera_id
            }
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        return {
            "status": "running",
            "camera_id": self.camera_id,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "target_fps": self.target_fps,
            "actual_fps": round(avg_fps, 2),
            "frame_count": self.frame_count,
            "uptime": round(elapsed, 1)
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    @staticmethod
    def list_cameras(max_cameras: int = 5) -> list[int]:
        """
        List available camera IDs.
        
        Args:
            max_cameras: Maximum number of cameras to check
            
        Returns:
            List of available camera IDs
        """
        available = []
        
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        
        return available

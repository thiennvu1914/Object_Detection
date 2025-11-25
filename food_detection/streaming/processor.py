"""
Frame Processor Module
======================
Queue-based frame processing with detection and encoding.
"""
import cv2
import numpy as np
import base64
from typing import Optional, Dict, Any
from queue import Queue, Empty
import threading
import time


class FrameProcessor:
    """
    Frame processor with queue management.
    
    Features:
    - Queue-based processing to avoid blocking
    - Frame skipping for performance
    - Base64 encoding for WebSocket
    - Annotated frame generation with bounding boxes
    """
    
    def __init__(
        self,
        pipeline,
        skip_frames: int = 2,
        max_queue_size: int = 2,
        encode_quality: int = 70,
        conf: float = 0.25
    ):
        """
        Initialize frame processor.
        
        Args:
            pipeline: FoodDetectionPipeline instance
            skip_frames: Process every Nth frame (1 = process all, 2 = skip 1, etc.)
            max_queue_size: Maximum frames in queue (keep small to avoid lag)
            encode_quality: JPEG encoding quality (1-100, lower=smaller file)
            conf: Detection confidence threshold (0.0-1.0)
        """
        self.pipeline = pipeline
        self.skip_frames = skip_frames
        self.max_queue_size = max_queue_size
        self.encode_quality = encode_quality
        self.conf = conf
        
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.is_running = False
        self.process_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.frames_processed = 0
        self.frames_skipped = 0
        self.frame_counter = 0
        
        # Latest result cache
        self.latest_result: Optional[Dict[str, Any]] = None
        self.result_lock = threading.Lock()
    
    def start(self):
        """Start processing thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        print(f"[FrameProcessor] Started (skip_frames={self.skip_frames})")
    
    def stop(self):
        """Stop processing thread."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        
        print(f"[FrameProcessor] Stopped (processed={self.frames_processed}, skipped={self.frames_skipped})")
    
    def submit_frame(self, frame: np.ndarray) -> bool:
        """
        Submit frame for processing.
        
        Args:
            frame: Frame as numpy array (BGR format)
            
        Returns:
            True if frame was queued, False if skipped
        """
        self.frame_counter += 1
        
        # Frame skipping logic
        if self.frame_counter % self.skip_frames != 0:
            self.frames_skipped += 1
            return False
        
        # If queue is full, clear old frames to prevent lag
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()  # Remove oldest frame
            except:
                pass
        
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except:
            # Queue is full, skip this frame
            self.frames_skipped += 1
            return False
    
    def _process_loop(self):
        """Internal processing loop (runs in separate thread)."""
        while self.is_running:
            try:
                # Get frame from queue (timeout to check is_running periodically)
                frame = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue
            
            try:
                # Save frame temporarily for detection
                import tempfile
                from pathlib import Path
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    cv2.imwrite(tmp.name, frame)
                    tmp_path = tmp.name
                
                # Run detection
                start_time = time.time()
                results = self.pipeline.process_image(tmp_path, conf=self.conf)
                processing_time = time.time() - start_time
                
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
                
                # Annotate frame with detection results
                annotated_frame = self._annotate_frame(frame.copy(), results['detections'])
                
                # Encode to base64
                base64_image = self._encode_frame(annotated_frame)
                
                # Update latest result
                with self.result_lock:
                    self.latest_result = {
                        'type': 'detection',
                        'image_base64': base64_image,
                        'detections': results['detections'],
                        'count': len(results['detections']),
                        'processing_time': round(processing_time, 3),
                        'timestamp': time.time()
                    }
                
                self.frames_processed += 1
                
            except Exception as e:
                print(f"[FrameProcessor] Error processing frame: {e}")
                # Update with error result
                with self.result_lock:
                    self.latest_result = {
                        'type': 'error',
                        'message': str(e),
                        'timestamp': time.time()
                    }
    
    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """
        Get latest processing result.
        
        Returns:
            Dict with detection results and annotated image, or None
        """
        with self.result_lock:
            return self.latest_result.copy() if self.latest_result else None
    
    def _annotate_frame(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.
        
        Args:
            frame: Original frame
            detections: List of detection dicts with bbox, class, score
            
        Returns:
            Annotated frame
        """
        # Define colors for different classes (BGR format)
        colors = {
            'coconut': (0, 255, 255),    # Yellow
            'cua': (255, 0, 0),          # Blue
            'macaron': (255, 0, 255),    # Magenta
            'meden': (0, 255, 0),        # Green
            'melon': (0, 165, 255),      # Orange
        }
        
        for det in detections:
            bbox = det['bbox']
            class_name = det['class']
            # Use similarity or confidence as score (pipeline returns 'similarity')
            score = det.get('similarity', det.get('confidence', det.get('score', 0.0)))
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get color for this class
            color = colors.get(class_name, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name} {score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # Add frame info
        info_text = f"Detected: {len(detections)} items"
        cv2.putText(
            frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return frame
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        """
        Encode frame to base64 JPEG.
        
        Args:
            frame: Frame as numpy array
            
        Returns:
            Base64 encoded string
        """
        # Resize frame to reduce bandwidth (optional, adjust as needed)
        # Keep aspect ratio, max width 640px
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Encode to JPEG with quality setting
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.encode_quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        # Convert to base64
        base64_str = base64.b64encode(buffer).decode('utf-8')
        
        return base64_str
    
    def get_stats(self) -> dict:
        """
        Get processing statistics.
        
        Returns:
            Dict with processing stats
        """
        return {
            'frames_processed': self.frames_processed,
            'frames_skipped': self.frames_skipped,
            'queue_size': self.frame_queue.qsize(),
            'skip_ratio': self.skip_frames
        }

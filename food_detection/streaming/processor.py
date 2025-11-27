"""
Frame Processor Module
======================
Queue-based frame processing with SSIM-based change detection optimization.

Features:
- Lightweight change detection layer (SSIM + Frame Difference)
- Only triggers YOLOE when significant changes detected
- Zero-training, menu-agnostic approach
- Queue-based processing to avoid blocking
"""
import cv2
import numpy as np
import base64
from typing import Optional, Dict, Any
from queue import Queue, Empty
import threading
import time
from food_detection.streaming.change_detector import ChangeDetector


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
        skip_frames: int = 15,
        max_queue_size: int = 1,
        encode_quality: int = 70,
        conf: float = 0.25,
        enable_change_detection: bool = True,
        change_detector: Optional[ChangeDetector] = None,
        auto_flush_queue: bool = True
    ):
        """
        Initialize frame processor.
        
        Args:
            pipeline: FoodDetectionPipeline instance
            skip_frames: Process every Nth frame (15 = ~2 FPS detection at 30 FPS camera)
                        Recommended: 10-20 for lag-free streaming
            max_queue_size: Maximum frames in queue (1 = no backlog, minimal lag)
            encode_quality: JPEG encoding quality (1-100, lower=smaller file)
            conf: Detection confidence threshold (0.0-1.0)
            enable_change_detection: Enable SSIM-based change detection layer
            change_detector: Custom ChangeDetector instance (or create default)
            auto_flush_queue: Automatically flush old frames to prevent backlog
        """
        self.pipeline = pipeline
        self.skip_frames = skip_frames
        self.max_queue_size = max_queue_size
        self.encode_quality = encode_quality
        self.conf = conf
        self.auto_flush_queue = auto_flush_queue
        
        # Change detection layer (pre-filter before YOLOE)
        self.enable_change_detection = enable_change_detection
        if enable_change_detection:
            self.change_detector = change_detector or ChangeDetector(
                ssim_threshold=0.80,      # 20% change triggers detection (less sensitive)
                diff_threshold=0.20,      # 20% pixels changed (less sensitive)
                resize_width=320          # Fast preprocessing
            )
            print("[FrameProcessor] Change detection enabled (SSIM + Frame Diff)")
            print(f"[FrameProcessor] Thresholds: SSIM={0.80}, Diff={0.20} (reduced sensitivity)")
        else:
            self.change_detector = None
            print("[FrameProcessor] Change detection disabled")
        
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.is_running = False
        self.process_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.frames_processed = 0
        self.frames_skipped = 0
        self.frames_skipped_by_change_detector = 0
        self.frames_flushed = 0  # Frames flushed to prevent backlog
        self.frame_counter = 0
        
        # Latest result cache
        self.latest_result: Optional[Dict[str, Any]] = None
        self.result_lock = threading.Lock()
        
        # Performance optimization
        detection_fps = 30.0 / skip_frames
        print(f"[FrameProcessor] Skip frames: {skip_frames} (~{detection_fps:.1f} FPS detection)")
        print(f"[FrameProcessor] Queue size: {max_queue_size} (auto_flush={auto_flush_queue})")
    
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
        
        print(f"[FrameProcessor] Stopped (processed={self.frames_processed}, skipped={self.frames_skipped}, flushed={self.frames_flushed})")
    
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
        
        # Auto-flush queue to prevent backlog (keep only latest frame)
        if self.auto_flush_queue:
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()  # Remove all old frames
                    self.frames_flushed += 1
                except:
                    break
        
        # If queue still full, force remove oldest frame
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
                self.frames_flushed += 1
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
                # ====== CHANGE DETECTION LAYER (Pre-filter) ======
                # Only trigger YOLOE if significant change detected
                should_detect = True
                change_metrics = None
                
                if self.enable_change_detection and self.change_detector:
                    should_detect, change_metrics = self.change_detector.detect_change(frame)
                    
                    if not should_detect:
                        # No significant change → Skip YOLOE and MobileCLIP
                        self.frames_skipped_by_change_detector += 1
                        
                        # Return cached result with change detection info
                        with self.result_lock:
                            if self.latest_result:
                                # Reuse previous detection result
                                cached_result = self.latest_result.copy()
                                cached_result['cached'] = True
                                cached_result['change_detection'] = change_metrics
                                cached_result['timestamp'] = time.time()
                                self.latest_result = cached_result
                        
                        continue  # Skip to next frame
                
                # ====== YOLOE + MobileCLIP PIPELINE ======
                # Change detected → Run full detection pipeline
                
                # Save frame temporarily for detection
                import tempfile
                from pathlib import Path
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    cv2.imwrite(tmp.name, frame)
                    tmp_path = tmp.name
                
                # Run detection (MobileCLIP only runs if YOLOE detects objects)
                start_time = time.time()
                results = self.pipeline.process_image(
                    tmp_path, 
                    conf=self.conf,
                    save_to_db=False  # Skip DB to reduce overhead
                )
                processing_time = time.time() - start_time
                
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
                
                # Annotate frame with detection results
                annotated_frame = self._annotate_frame(frame.copy(), results['detections'])
                
                # Encode to base64
                base64_image = self._encode_frame(annotated_frame)
                
                # Update latest result (include change detection metrics)
                with self.result_lock:
                    self.latest_result = {
                        'type': 'detection',
                        'image_base64': base64_image,
                        'detections': results['detections'],
                        'count': len(results['detections']),
                        'processing_time': round(processing_time, 3),
                        'cached': False,
                        'change_detection': change_metrics,
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
            Dict with processing stats including change detection efficiency
        """
        stats = {
            'frames_processed': self.frames_processed,
            'frames_skipped': self.frames_skipped,
            'frames_skipped_by_change_detector': self.frames_skipped_by_change_detector,
            'frames_flushed': self.frames_flushed,
            'queue_size': self.frame_queue.qsize(),
            'skip_ratio': self.skip_frames
        }
        
        # Add change detection statistics if enabled
        if self.enable_change_detection and self.change_detector:
            change_stats = self.change_detector.get_statistics()
            stats['change_detection'] = change_stats
            
            # Calculate total optimization ratio (skip_frames + change detection)
            total_frames_captured = self.frame_counter
            yoloe_calls = self.frames_processed
            
            if total_frames_captured > 0:
                # Frames that reached processor after skip_frames
                frames_submitted = total_frames_captured // self.skip_frames
                
                # Combined optimization
                skip_frames_reduction = 1.0 - (1.0 / self.skip_frames)
                
                if frames_submitted > 0:
                    change_detection_reduction = self.frames_skipped_by_change_detector / frames_submitted
                    total_reduction = skip_frames_reduction + (1.0 - skip_frames_reduction) * change_detection_reduction
                else:
                    total_reduction = skip_frames_reduction
                
                stats['optimization_ratio'] = f"{total_reduction*100:.1f}%"
                stats['effective_fps'] = f"{(yoloe_calls / total_frames_captured * 30.0):.2f} FPS"
        
        return stats

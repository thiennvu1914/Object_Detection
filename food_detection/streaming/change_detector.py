"""
Lightweight Change Detection Layer
===================================
Zero-training change detection using SSIM + Frame Difference.
Acts as a pre-filter before YOLOE to optimize performance.

Key features:
- Structural Similarity Index (SSIM) for perceptual changes
- Frame difference for motion detection
- Configurable threshold (0.15-0.20 recommended)
- Menu-agnostic (works regardless of food items)
- GPU-accelerated when available
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from skimage.metrics import structural_similarity as ssim
import time


class ChangeDetector:
    """
    Lightweight change detector for video frames.
    
    Uses SSIM + frame difference to detect meaningful changes.
    Only triggers downstream processing when change threshold is exceeded.
    """
    
    def __init__(
        self,
        ssim_threshold: float = 0.85,
        diff_threshold: float = 0.08,
        enable_blur: bool = False,
        blur_kernel: Tuple[int, int] = (3, 3),
        resize_height: int = 360,
        resize_width: int = None,  # Deprecated, kept for backward compatibility
        roi: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) in original frame coords
    ):
        """
        Initialize change detector.
        
        Args:
            ssim_threshold: SSIM similarity threshold (0.0-1.0)
                          Lower = more sensitive to changes
                          Typical: 0.85 (means 15% change triggers detection)
                          For ROI-based: 0.92-0.95 (high sensitivity for small changes)
            diff_threshold: Frame difference threshold (0.0-1.0)
                          Percentage of pixels that must change
                          Typical: 0.08-0.12 for 360p-480p resolution
                          For ROI-based: 0.03-0.06 (detect even small item changes)
            enable_blur: Apply Gaussian blur to reduce noise (default: False for better edge detection)
            blur_kernel: Blur kernel size (must be odd)
            resize_height: Resize frame HEIGHT for faster processing (better for top-down cameras)
                         Maintains aspect ratio. Default: 360 (good balance speed/accuracy)
            resize_width: Deprecated. Use resize_height instead.
            roi: Region of Interest (x1, y1, x2, y2) in original frame coordinates.
                 If provided, only this region will be analyzed for changes.
                 Useful for focusing on tray area and ignoring background.
        """
        self.ssim_threshold = ssim_threshold
        self.diff_threshold = diff_threshold
        self.enable_blur = enable_blur
        self.blur_kernel = blur_kernel
        self.roi = roi
        
        # Backward compatibility: if resize_width provided, use it as resize_height
        if resize_width is not None:
            self.resize_height = resize_width
            print(f"[Warning] resize_width is deprecated. Using resize_height={resize_width} instead.")
        else:
            self.resize_height = resize_height
        
        # State
        self.prev_frame = None
        self.frame_count = 0
        self.change_count = 0
        self.no_change_count = 0
        
        # Statistics
        self.total_processing_time = 0.0
        
        roi_str = f"ROI={roi}" if roi else "Full Frame"
        print(f"[ChangeDetector] Initialized with SSIM threshold: {ssim_threshold}, "
              f"Diff threshold: {diff_threshold}, {roi_str}")
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for change detection.
        
        Steps:
        1. Extract ROI if specified (focus on tray area)
        2. Resize to fixed HEIGHT (better for top-down cameras)
        3. Convert to grayscale
        4. Apply Gaussian blur (optional, reduce noise)
        
        Args:
            frame: Input frame (BGR, any size)
            
        Returns:
            Preprocessed grayscale frame (resized)
        """
        # Extract ROI if specified (focus on tray/food area only)
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            frame = frame[y1:y2, x1:x2]
        
        # Resize to fixed HEIGHT (maintain aspect ratio)
        # This is better for top-down cameras where width > height
        height, width = frame.shape[:2]
        if height > self.resize_height:
            scale = self.resize_height / height
            new_width = int(width * scale)
            frame = cv2.resize(frame, (new_width, self.resize_height), 
                             interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Apply Gaussian blur to reduce noise (usually disabled for better edge detection)
        if self.enable_blur:
            gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        
        return gray
    
    def _calculate_ssim(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index between two frames.
        
        SSIM measures perceptual similarity (0.0-1.0):
        - 1.0: Identical frames
        - 0.85: 15% structural change (typical threshold)
        - 0.7: 30% structural change (significant)
        - <0.5: Very different frames
        
        Args:
            frame1: First grayscale frame
            frame2: Second grayscale frame
            
        Returns:
            SSIM score (0.0-1.0)
        """
        # Compute SSIM with default parameters
        # win_size=7 is default for structural similarity
        score, _ = ssim(frame1, frame2, full=True)
        return score
    
    def _calculate_frame_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate frame difference percentage.
        
        Measures how many pixels changed between frames.
        
        Args:
            frame1: First grayscale frame
            frame2: Second grayscale frame
            
        Returns:
            Difference ratio (0.0-1.0)
        """
        # Calculate absolute difference
        diff = cv2.absdiff(frame1, frame2)
        
        # Threshold to binary (pixel changed if diff > 25)
        # Lower threshold = more sensitive to small changes
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of changed pixels
        changed_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.size
        diff_ratio = changed_pixels / total_pixels
        
        return diff_ratio
    
    def detect_change(self, frame: np.ndarray, force_detect: bool = False) -> Tuple[bool, dict]:
        """
        Detect if significant change occurred between current and previous frame.
        
        Detection logic:
        1. If first frame → always detect (no comparison possible)
        2. Calculate SSIM and frame difference
        3. If SSIM < threshold OR diff > threshold → CHANGE DETECTED
        4. Otherwise → NO CHANGE (skip detection)
        
        Args:
            frame: Current frame (BGR, full resolution)
            force_detect: If True, always return True (bypass detection)
            
        Returns:
            Tuple of (should_detect, metrics):
            - should_detect: True if change detected, False otherwise
            - metrics: Dict with SSIM score, diff ratio, processing time
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Force detection if requested (e.g., manual trigger)
        if force_detect:
            self.prev_frame = self._preprocess_frame(frame)
            self.change_count += 1
            return True, {
                'ssim': 0.0,
                'diff_ratio': 1.0,
                'processing_time': 0.0,
                'reason': 'forced'
            }
        
        # First frame: always detect (no previous frame to compare)
        if self.prev_frame is None:
            self.prev_frame = self._preprocess_frame(frame)
            self.change_count += 1
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            return True, {
                'ssim': 0.0,
                'diff_ratio': 1.0,
                'processing_time': processing_time,
                'reason': 'first_frame'
            }
        
        # Preprocess current frame
        curr_frame = self._preprocess_frame(frame)
        
        # Calculate SSIM (structural similarity)
        ssim_score = self._calculate_ssim(self.prev_frame, curr_frame)
        
        # Calculate frame difference
        diff_ratio = self._calculate_frame_diff(self.prev_frame, curr_frame)
        
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        
        # Decision logic: detect change if SSIM low OR diff high
        # SSIM < threshold means significant structural change
        # diff > threshold means significant pixel movement
        change_detected = (ssim_score < self.ssim_threshold) or (diff_ratio > self.diff_threshold)
        
        if change_detected:
            self.change_count += 1
            reason = []
            if ssim_score < self.ssim_threshold:
                reason.append(f'ssim={ssim_score:.3f}<{self.ssim_threshold}')
            if diff_ratio > self.diff_threshold:
                reason.append(f'diff={diff_ratio:.3f}>{self.diff_threshold}')
            reason_str = ', '.join(reason)
        else:
            self.no_change_count += 1
            reason_str = f'ssim={ssim_score:.3f}, diff={diff_ratio:.3f} (stable)'
        
        # Update previous frame for next comparison
        self.prev_frame = curr_frame
        
        metrics = {
            'ssim': float(ssim_score),
            'diff_ratio': float(diff_ratio),
            'processing_time': processing_time,
            'reason': reason_str,
            'frame_count': self.frame_count,
            'change_count': self.change_count,
            'no_change_count': self.no_change_count
        }
        
        return change_detected, metrics
    
    def reset(self):
        """Reset detector state (clears previous frame)."""
        self.prev_frame = None
        self.frame_count = 0
        self.change_count = 0
        self.no_change_count = 0
        self.total_processing_time = 0.0
        print("[ChangeDetector] Reset")
    
    def get_statistics(self) -> dict:
        """
        Get detection statistics.
        
        Returns:
            Dict with statistics:
            - frame_count: Total frames processed
            - change_count: Frames with changes detected
            - no_change_count: Frames skipped (no change)
            - skip_ratio: Percentage of frames skipped
            - avg_processing_time: Average time per frame (ms)
        """
        skip_ratio = self.no_change_count / self.frame_count if self.frame_count > 0 else 0.0
        avg_time = (self.total_processing_time / self.frame_count * 1000) if self.frame_count > 0 else 0.0
        
        return {
            'frame_count': self.frame_count,
            'change_count': self.change_count,
            'no_change_count': self.no_change_count,
            'skip_ratio': skip_ratio,
            'avg_processing_time_ms': avg_time,
            'total_processing_time': self.total_processing_time
        }
    
    def print_statistics(self):
        """Print detection statistics to console."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("Change Detection Statistics")
        print("="*60)
        print(f"Total frames processed:    {stats['frame_count']}")
        print(f"Changes detected:          {stats['change_count']} ({stats['change_count']/stats['frame_count']*100:.1f}%)")
        print(f"Frames skipped (no change): {stats['no_change_count']} ({stats['skip_ratio']*100:.1f}%)")
        print(f"Avg processing time:       {stats['avg_processing_time_ms']:.2f} ms/frame")
        print(f"Total processing time:     {stats['total_processing_time']:.2f} s")
        print("="*60 + "\n")


class AdaptiveChangeDetector(ChangeDetector):
    """
    Adaptive change detector with dynamic threshold adjustment.
    
    Automatically adjusts thresholds based on scene stability:
    - In stable scenes (low motion): Higher sensitivity (detect small changes)
    - In dynamic scenes (high motion): Lower sensitivity (avoid false triggers)
    """
    
    def __init__(
        self,
        initial_ssim_threshold: float = 0.85,
        initial_diff_threshold: float = 0.15,
        adaptation_rate: float = 0.1,
        min_ssim_threshold: float = 0.75,
        max_ssim_threshold: float = 0.95,
        **kwargs
    ):
        """
        Initialize adaptive change detector.
        
        Args:
            initial_ssim_threshold: Starting SSIM threshold
            initial_diff_threshold: Starting diff threshold
            adaptation_rate: How fast to adapt (0.0-1.0)
            min_ssim_threshold: Minimum SSIM threshold (most sensitive)
            max_ssim_threshold: Maximum SSIM threshold (least sensitive)
            **kwargs: Additional arguments for base ChangeDetector
        """
        super().__init__(
            ssim_threshold=initial_ssim_threshold,
            diff_threshold=initial_diff_threshold,
            **kwargs
        )
        
        self.initial_ssim = initial_ssim_threshold
        self.initial_diff = initial_diff_threshold
        self.adaptation_rate = adaptation_rate
        self.min_ssim = min_ssim_threshold
        self.max_ssim = max_ssim_threshold
        
        # Adaptive state
        self.recent_ssim_scores = []
        self.recent_diff_ratios = []
        self.history_size = 10  # Keep last N measurements
        
        print(f"[AdaptiveChangeDetector] Adaptive thresholds enabled "
              f"(range: {min_ssim_threshold}-{max_ssim_threshold})")
    
    def detect_change(self, frame: np.ndarray, force_detect: bool = False) -> Tuple[bool, dict]:
        """
        Detect change with adaptive thresholds.
        
        Overrides base detect_change to add adaptive behavior.
        """
        # Call base detection
        change_detected, metrics = super().detect_change(frame, force_detect)
        
        if not force_detect and metrics['reason'] not in ['first_frame', 'forced']:
            # Update history
            self.recent_ssim_scores.append(metrics['ssim'])
            self.recent_diff_ratios.append(metrics['diff_ratio'])
            
            # Keep only recent history
            if len(self.recent_ssim_scores) > self.history_size:
                self.recent_ssim_scores.pop(0)
                self.recent_diff_ratios.pop(0)
            
            # Adapt thresholds based on scene stability
            if len(self.recent_ssim_scores) >= 5:
                avg_ssim = np.mean(self.recent_ssim_scores)
                avg_diff = np.mean(self.recent_diff_ratios)
                
                # If scene is stable (high SSIM, low diff) → increase sensitivity
                if avg_ssim > 0.9 and avg_diff < 0.1:
                    target_ssim = min(self.ssim_threshold + 0.02, self.max_ssim)
                # If scene is dynamic (low SSIM, high diff) → decrease sensitivity
                elif avg_ssim < 0.8 or avg_diff > 0.2:
                    target_ssim = max(self.ssim_threshold - 0.02, self.min_ssim)
                else:
                    target_ssim = self.initial_ssim
                
                # Smooth adaptation
                self.ssim_threshold += (target_ssim - self.ssim_threshold) * self.adaptation_rate
        
        return change_detected, metrics

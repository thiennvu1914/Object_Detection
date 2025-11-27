"""
Change Detection Layer Demo
============================
Standalone test script for SSIM-based change detection.

Usage:
    python test_change_detection.py

Features:
- Tests ChangeDetector with webcam
- Shows change detection metrics in real-time
- Displays optimization statistics
"""
import cv2
import numpy as np
import time
from food_detection.streaming.change_detector import ChangeDetector


def main():
    """Run change detection demo with webcam."""
    print("\n" + "="*60)
    print("Change Detection Layer Demo")
    print("="*60)
    print("Press 'q' to quit")
    print("Press 'r' to reset detector")
    print("Press 's' to show statistics")
    print("="*60 + "\n")
    
    # Initialize change detector
    detector = ChangeDetector(
        ssim_threshold=0.85,     # 15% change triggers detection
        diff_threshold=0.15,     # 15% pixels changed
        enable_blur=True,
        resize_width=320
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("[INFO] Webcam opened successfully")
    print("[INFO] Change detection running...")
    
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break
        
        # Run change detection
        change_detected, metrics = detector.detect_change(frame)
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        # Create display frame
        display = frame.copy()
        
        # Draw status indicator
        if change_detected:
            # Green = Change detected (YOLOE would run)
            status_color = (0, 255, 0)
            status_text = "CHANGE DETECTED - YOLOE TRIGGERED"
            circle_color = (0, 255, 0)
        else:
            # Red = No change (YOLOE skipped)
            status_color = (0, 0, 255)
            status_text = "NO CHANGE - YOLOE SKIPPED"
            circle_color = (0, 0, 255)
        
        # Draw status box
        cv2.rectangle(display, (10, 10), (630, 150), (0, 0, 0), -1)
        cv2.rectangle(display, (10, 10), (630, 150), status_color, 2)
        
        # Draw status indicator circle
        cv2.circle(display, (30, 35), 12, circle_color, -1)
        
        # Draw text
        cv2.putText(display, status_text, (50, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Draw metrics
        if metrics:
            y_offset = 70
            cv2.putText(display, f"SSIM: {metrics['ssim']:.3f} (threshold: 0.85)",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(display, f"Diff Ratio: {metrics['diff_ratio']:.3f} (threshold: 0.15)",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(display, f"Processing: {metrics['processing_time']*1000:.1f} ms",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw FPS and frame count
        cv2.putText(display, f"FPS: {current_fps:.1f}", (540, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(display, f"Frames: {metrics.get('frame_count', 0)}", (540, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(display, f"Changes: {metrics.get('change_count', 0)}", (540, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(display, f"Skipped: {metrics.get('no_change_count', 0)}", (540, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw instructions
        cv2.putText(display, "Press 'q' to quit, 'r' to reset, 's' for stats",
                   (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show frame
        cv2.imshow('Change Detection Demo', display)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset()
            print("\n[INFO] Detector reset")
        elif key == ord('s'):
            detector.print_statistics()
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n" + "="*60)
    print("Final Statistics")
    print("="*60)
    detector.print_statistics()
    
    stats = detector.get_statistics()
    if stats['frame_count'] > 0:
        print(f"\nOptimization Benefit:")
        print(f"  - YOLOE calls saved: {stats['no_change_count']} / {stats['frame_count']}")
        print(f"  - Performance gain: {stats['skip_ratio']*100:.1f}% fewer YOLOE calls")
        print(f"  - Avg detection time: {stats['avg_processing_time_ms']:.2f} ms")
        print(f"    (vs YOLOE ~1.3s = {1300/stats['avg_processing_time_ms']:.0f}x faster)")


if __name__ == '__main__':
    main()

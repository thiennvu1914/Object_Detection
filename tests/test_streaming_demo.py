"""
Streaming Demo with Change Detection (Optimized UI + Performance)
=================================================================
Real-time food detection with SSIM-based change detection layer.

Camera LEFT  |  Stats RIGHT
"""

import cv2
import numpy as np
import time
import tempfile
from pathlib import Path

from food_detection.core.pipeline import FoodDetectionPipeline
from food_detection.streaming.camera import CameraCapture
from food_detection.streaming.change_detector import ChangeDetector


# ===============================================================
#  STATS PANEL (RIGHT SIDE)
# ===============================================================

def build_stats_panel(info, width=400, height=480):
    """Create right-side statistics panel with detailed detection info."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    white = (255, 255, 255)
    yellow = (0, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)

    y = 25
    dy = 25

    # Title
    cv2.putText(panel, "=== DETECTION STATS ===", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, yellow, 2)
    y += dy + 5

    # Performance metrics
    cv2.putText(panel, f"FPS: {info['fps']:.1f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1); y += dy
    
    cv2.putText(panel, f"Frame: #{info['frame_count']}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1); y += dy

    cv2.putText(panel, f"YOLOE calls: {info['yoloe_calls']}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, green, 1); y += dy

    cv2.putText(panel, f"Frames skipped: {info['skipped']}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1); y += dy
    
    cv2.putText(panel, f"Optimization: {info['skip_ratio']:.1f}%", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, yellow, 1); y += dy

    y += 5
    cv2.line(panel, (10, y), (width-10, y), white, 1)
    y += 15

    # Change detection status (fixed height area)
    cv2.putText(panel, "Change Detection:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1); y += dy

    status_color = green if info['change_detected'] else red
    status_text = "CHANGE DETECTED" if info['change_detected'] else "NO CHANGE"
    cv2.putText(panel, status_text, (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2); y += dy

    # Always show SSIM and Diff (fixed positions)
    if info['change_metrics']:
        ssim_val = info['change_metrics']['ssim']
        diff_val = info['change_metrics']['diff_ratio']
    else:
        ssim_val = 0.0
        diff_val = 0.0
    
    cv2.putText(panel, f"  SSIM: {ssim_val:.3f} (th: 0.94)",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1); y += dy-3
    cv2.putText(panel, f"  Diff: {diff_val:.3f} (th: 0.05)",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1); y += dy-3

    y += 5
    cv2.line(panel, (10, y), (width-10, y), white, 1)
    y += 15

    # Detected objects
    cv2.putText(panel, "Detected Objects:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1); y += dy

    # Fixed area for detections (reserve space for max 5 classes)
    detection_area_y = y
    max_classes = 5
    
    if info['detections']:
        # Count by class
        class_counts = {}
        class_scores = {}
        for det in info['detections']:
            cls = det['class']
            score = det.get('similarity', det.get('confidence', 0.0))
            class_counts[cls] = class_counts.get(cls, 0) + 1
            if cls not in class_scores:
                class_scores[cls] = []
            class_scores[cls].append(score)
        
        # Display summary
        colors = {
            'coconut': (0, 255, 255),
            'cua': (255, 0, 0),
            'macaron': (255, 0, 255),
            'meden': (0, 255, 0),
            'melon': (0, 165, 255),
        }
        
        # Sort classes for consistent ordering
        for cls in sorted(class_counts.keys()):
            count = class_counts[cls]
            avg_score = sum(class_scores[cls]) / len(class_scores[cls])
            color = colors.get(cls, green)
            
            cv2.putText(panel, f"  {cls}: {count}x", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(panel, f"({avg_score:.2f})", (width-100, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
            y += dy-5
        
        # Skip to end of reserved area
        y = detection_area_y + (max_classes * (dy-5))
        
        y += 5
        cv2.putText(panel, f"Total: {len(info['detections'])} items", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, yellow, 1); y += dy
    else:
        cv2.putText(panel, "  No objects detected", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        # Skip to end of reserved area
        y = detection_area_y + (max_classes * (dy-5)) + 5 + dy

    y += 5
    cv2.line(panel, (10, y), (width-10, y), white, 1)
    y += 15

    # Controls
    cv2.putText(panel, "Controls:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1); y += dy
    
    controls = [
        ("'q'/'ESC'", "Quit"),
        ("'r'", "Reset detector"),
        ("'p'", "Pause/Resume"),
        ("'+'", "Speed up detection"),
        ("'-'", "Slow down detection"),
    ]
    
    for key, desc in controls:
        cv2.putText(panel, f"  {key}: {desc}", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, white, 1); y += dy-8

    return panel


# ===============================================================
#  DRAW DETECTIONS
# ===============================================================

def draw_detections(frame, detections):
    """Draw bounding boxes and labels."""
    colors = {
        'coconut': (0, 255, 255),
        'cua': (255, 0, 0),
        'macaron': (255, 0, 255),
        'meden': (0, 255, 0),
        'melon': (0, 165, 255),
    }

    for det in detections:
        bbox = det['bbox']
        cls = det['class']
        score = det.get('similarity', det.get('confidence', 0.0))

        x1, y1, x2, y2 = map(int, bbox)
        color = colors.get(cls, (0, 255, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{cls} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


# ===============================================================
#  MAIN STREAMING LOOP
# ===============================================================

def main():

    print("\n===============================")
    print(" FOOD DETECTION DEMO (OPTIMIZED)")
    print("===============================\n")

    # Start camera FIRST (faster feedback)
    print("Starting camera...")
    camera = CameraCapture(camera_id=0, width=640, height=480, fps=30)
    if not camera.start():
        print("✗ Cannot open camera")
        return
    print("✓ Camera started")

    # Change detector (lightweight, instant)
    # ROI-based detection: Focus on tray area (center region where food items are placed)
    # For 640x480 camera: use center 70% area to ignore edges/background
    frame_width = 640
    frame_height = 480
    roi_margin_x = int(frame_width * 0.15)  # 15% margin on sides
    roi_margin_y = int(frame_height * 0.15)  # 15% margin top/bottom
    roi = (roi_margin_x, roi_margin_y, frame_width - roi_margin_x, frame_height - roi_margin_y)
    
    change_detector = ChangeDetector(
        ssim_threshold=0.94,   # HIGH sensitivity for small item changes (0.92-0.96)
        diff_threshold=0.05,   # Detect even 5% pixel change in ROI (0.04-0.06)
        resize_height=360,     # Resize by HEIGHT (better for top-down camera)
        enable_blur=False,     # Disable blur to preserve edges and details
        roi=roi                # Focus on tray area only (96, 72, 544, 408)
    )
    print(f"✓ ChangeDetector initialized (ROI-Based High Sensitivity)")
    print(f"  → ROI: {roi} (center 70% area)")
    print(f"  → SSIM: 0.94, Diff: 0.05 (optimized for item add/remove)")

    # Load model LAST (heavy operation, but camera already running)
    print("\nLoading AI models (YOLOE + MobileCLIP)...")
    load_start = time.time()
    pipeline = FoodDetectionPipeline()
    load_time = time.time() - load_start
    print(f"✓ Models loaded in {load_time:.1f}s\n")

    # --- Streaming State ---
    frame_count = 0
    yoloe_calls = 0
    skipped = 0
    last_detections = []

    fps_start = time.time()
    fps_counter = 0
    current_fps = 0.0

    paused = False
    detection_interval = 5  # process every 5th frame

    print("[INFO] DetectionInterval:", detection_interval)

    # Create resizable window (not fullscreen)
    window_name = "Food Detection Streaming Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)  # Default size, can be resized by user

    # MAIN LOOP -----------------------------------------------------
    while True:

        frame = camera.get_frame()
        if frame is None:
            continue

        frame_count += 1
        fps_counter += 1

        # FPS tracking
        if time.time() - fps_start >= 1.0:
            current_fps = fps_counter / (time.time() - fps_start)
            fps_counter = 0
            fps_start = time.time()

        display = frame.copy()
        
        # Draw ROI rectangle on display (semi-transparent green box)
        roi_color = (0, 255, 0)  # Green
        cv2.rectangle(display, (roi[0], roi[1]), (roi[2], roi[3]), roi_color, 2)
        cv2.putText(display, "ROI: Tray Area", (roi[0]+5, roi[1]+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)

        change_detected = False
        change_metrics = None

        # Process every N frames
        if not paused and (frame_count % detection_interval == 0):

            change_detected, change_metrics = change_detector.detect_change(frame)

            if change_detected:
                # RUN YOLOE + CLIP
                start = time.time()
                
                # Save frame temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    cv2.imwrite(tmp.name, frame)
                    tmp_path = tmp.name
                
                try:
                    results = pipeline.process_image(
                        tmp_path,
                        conf=0.25,
                        save_to_db=False
                    )
                    last_detections = results["detections"]
                    yoloe_calls += 1
                    print(f"[Frame {frame_count}] Detected {len(last_detections)} items in {time.time()-start:.2f}s")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

            else:
                skipped += 1

        else:
            skipped += 1

        # Draw detections
        if last_detections:
            display = draw_detections(display, last_detections)

        # Stats panel
        stats = {
            'change_detected': change_detected and not paused,
            'change_metrics': change_metrics,
            'detections': last_detections,
            'fps': current_fps,
            'frame_count': frame_count,
            'yoloe_calls': yoloe_calls,
            'skipped': skipped,
            'skip_ratio': skipped / frame_count * 100
        }
        panel = build_stats_panel(stats, height=display.shape[0])

        # Combine windows
        combined = np.hstack((display, panel))

        cv2.imshow(window_name, combined)

        # KEYBOARD HANDLERS ------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # ESC or q to exit
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('+'):
            detection_interval = max(1, detection_interval - 1)
        elif key == ord('-'):
            detection_interval = min(30, detection_interval + 1)
        elif key == ord('r'):
            change_detector.reset()
            skipped = 0
            yoloe_calls = 0
            last_detections = []


    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()

    print("\n===== FINAL STATS =====")
    print("Frames:", frame_count)
    print("YOLOE calls:", yoloe_calls)
    print("Skipped:", skipped)
    print("========================\n")


# ===============================================================

if __name__ == "__main__":
    main()

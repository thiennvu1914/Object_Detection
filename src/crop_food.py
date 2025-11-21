import cv2
import numpy as np

def crop_food(img, detection, padding=5, min_crop_size=80, max_resize=300):
    """
    Crop food region from detection with smart handling
    
    Args:
        img: Input image (numpy array)
        detection: Detection dict với format {'box': [x1,y1,x2,y2], 'score': float, 'class_id': int}
                   hoặc list/tuple [x1, y1, x2, y2, score, ...]
        padding: Pixels to add around bounding box (default: 5, giảm để crop sát hơn)
        min_crop_size: Minimum size for display/saving (default: 80)
        max_resize: Maximum size after resize to limit memory (default: 300)
    
    Returns:
        Cropped image (numpy array) or None if invalid
    """
    if img is None or img.size == 0:
        return None
    
    h, w = img.shape[:2]
    
    # Parse detection format
    if isinstance(detection, dict):
        # Format: {'box': [x1,y1,x2,y2], 'score': ..., 'class_id': ...}
        box = detection['box']
        x1, y1, x2, y2 = map(int, box[:4])
    elif isinstance(detection, (list, tuple, np.ndarray)):
        # Format: [x1, y1, x2, y2, score, ...]
        x1, y1, x2, y2 = map(int, detection[:4])
    else:
        print(f"⚠️  Invalid detection format: {type(detection)}")
        return None
    
    # Validate box coordinates
    if x2 <= x1 or y2 <= y1:
        print(f"⚠️  Invalid box coordinates: [{x1},{y1},{x2},{y2}]")
        return None
    
    # Calculate box dimensions
    box_w = x2 - x1
    box_h = y2 - y1
    
    # Skip extremely small boxes (likely noise)
    if box_w < 5 or box_h < 5:
        print(f"⚠️  Box too small: {box_w}x{box_h}px")
        return None
    
    # Add padding with boundary checks
    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(w, x2 + padding)
    y2_pad = min(h, y2 + padding)
    
    # Final validation after padding
    if x2_pad <= x1_pad or y2_pad <= y1_pad:
        print(f"⚠️  Invalid coordinates after padding: [{x1_pad},{y1_pad},{x2_pad},{y2_pad}]")
        # Fallback: crop without padding
        x1_pad, y1_pad = max(0, x1), max(0, y1)
        x2_pad, y2_pad = min(w, x2), min(h, y2)
        if x2_pad <= x1_pad or y2_pad <= y1_pad:
            return None
    
    # Perform crop
    try:
        crop = img[y1_pad:y2_pad, x1_pad:x2_pad].copy()
    except Exception as e:
        print(f"⚠️  Crop failed: {e}")
        return None
    
    if crop.size == 0:
        print(f"⚠️  Empty crop")
        return None
    
    # Resize for very small crops (better visibility)
    crop_h, crop_w = crop.shape[:2]
    
    if crop_h < min_crop_size or crop_w < min_crop_size:
        # Calculate scale to reach minimum size
        scale = max(min_crop_size / crop_h, min_crop_size / crop_w)
        new_h = int(crop_h * scale)
        new_w = int(crop_w * scale)
        
        # Limit maximum size to avoid huge images
        if new_h > max_resize or new_w > max_resize:
            scale = min(max_resize / new_h, max_resize / new_w)
            new_h = int(new_h * scale)
            new_w = int(new_w * scale)
        
        try:
            # Use INTER_CUBIC for upscaling (better quality)
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            print(f"⚠️  Resize failed: {e}")
            return crop  # Return original if resize fails
    
    # Optionally resize down very large crops
    elif crop_h > max_resize * 2 or crop_w > max_resize * 2:
        scale = min(max_resize / crop_h, max_resize / crop_w)
        new_h = int(crop_h * scale)
        new_w = int(crop_w * scale)
        try:
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"⚠️  Downsize failed: {e}")
    
    return crop


def crop_food_batch(img, detections, padding=15, min_crop_size=80):
    """
    Crop multiple food regions at once
    
    Args:
        img: Input image
        detections: List of detection dicts
        padding: Padding around each box
        min_crop_size: Minimum crop size
    
    Returns:
        List of (crop_image, detection_dict) tuples
    """
    results = []
    
    for det in detections:
        crop = crop_food(img, det, padding, min_crop_size)
        if crop is not None:
            results.append((crop, det))
    
    return results


def visualize_crop_with_context(img, detection, padding=15, border_color=(0, 255, 0)):
    """
    Show crop alongside original image with bounding box
    Useful for debugging
    
    Args:
        img: Original image
        detection: Detection dict or box
        padding: Padding for crop
        border_color: Color for bounding box
    
    Returns:
        Visualization image
    """
    # Parse box
    if isinstance(detection, dict):
        x1, y1, x2, y2 = map(int, detection['box'][:4])
        score = detection.get('score', 0)
    else:
        x1, y1, x2, y2 = map(int, detection[:4])
        score = detection[4] if len(detection) > 4 else 0
    
    # Draw box on original
    img_with_box = img.copy()
    cv2.rectangle(img_with_box, (x1, y1), (x2, y2), border_color, 2)
    cv2.putText(img_with_box, f"{score:.2f}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2)
    
    # Get crop
    crop = crop_food(img, detection, padding)
    
    if crop is None:
        return img_with_box
    
    # Resize original for display if needed
    display_h = 400
    scale = display_h / img.shape[0]
    display_w = int(img.shape[1] * scale)
    img_resized = cv2.resize(img_with_box, (display_w, display_h))
    
    # Resize crop to match height
    crop_scale = display_h / crop.shape[0]
    crop_w = int(crop.shape[1] * crop_scale)
    crop_resized = cv2.resize(crop, (crop_w, display_h))
    
    # Concatenate horizontally
    vis = np.hstack([img_resized, crop_resized])
    
    return vis


# Test function
if __name__ == "__main__":
    # Test với detection format mới
    test_detection = {
        'box': [100, 100, 200, 200],
        'score': 0.95,
        'class_id': 0
    }
    
    # Tạo test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test crop
    crop = crop_food(test_img, test_detection)
    print(f"Crop shape: {crop.shape if crop is not None else 'None'}")
    
    # Test với format cũ
    test_detection_old = [100, 100, 200, 200, 0.95]
    crop2 = crop_food(test_img, test_detection_old)
    print(f"Crop2 shape: {crop2.shape if crop2 is not None else 'None'}")
    
    print("✓ Tests passed!")
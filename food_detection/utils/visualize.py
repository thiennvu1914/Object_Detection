"""
Visualization utilities
=======================
Draw detection results on images.
"""
import cv2
import numpy as np
from typing import List, Dict


# Fixed colors for base classes (BGR format)
BASE_CLASS_COLORS = {
    'coconut': (255, 100, 0),      # Orange-blue
    'cua': (0, 100, 255),          # Orange
    'macaron': (0, 255, 100),      # Green
    'meden': (100, 0, 255),        # Purple
    'melon': (0, 200, 200),        # Yellow
}

# Cache for dynamically generated colors
_dynamic_colors = {}


def get_class_color(class_name: str, seed: int = 42):
    """
    Get consistent color for a class name.
    
    Args:
        class_name: Name of the class
        seed: Random seed for color generation
        
    Returns:
        BGR color tuple
    """
    if class_name in BASE_CLASS_COLORS:
        return BASE_CLASS_COLORS[class_name]
    
    if class_name not in _dynamic_colors:
        # Generate consistent color from hash
        np.random.seed(abs(hash(class_name)) % (2**32) + seed)
        color = tuple(np.random.randint(50, 255, 3).tolist())
        _dynamic_colors[class_name] = color
    
    return _dynamic_colors[class_name]


def visualize_detections(
    image: np.ndarray,
    detections: List[Dict],
    show_confidence: bool = True,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw detection results on image.
    
    Args:
        image: Input image (BGR)
        detections: List of detection dicts with bbox, class, similarity
        show_confidence: Whether to show confidence scores
        thickness: Line thickness for bounding boxes
        
    Returns:
        Image with drawn detections
    """
    result = image.copy()
    
    for det in detections:
        bbox = det['bbox']
        class_name = det['class']
        similarity = det.get('similarity', 0.0)
        
        # Get color for class
        color = get_class_color(class_name)
        
        # Draw bounding box
        x1, y1, x2, y2 = [int(c) for c in bbox]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        if show_confidence:
            label = f"{class_name} ({similarity:.2f})"
        else:
            label = class_name
        
        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        cv2.rectangle(
            result,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            result,
            label,
            (x1, y1 - baseline - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )
    
    return result

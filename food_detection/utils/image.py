"""
Image utilities
===============
Helper functions for image processing.
"""
import cv2
import numpy as np
from pathlib import Path


def save_crop(image: np.ndarray, bbox: list, output_path: str, padding: int = 5):
    """
    Save cropped region from image.
    
    Args:
        image: Input image array
        bbox: Bounding box [x1, y1, x2, y2]
        output_path: Path to save cropped image
        padding: Padding pixels around bbox
    """
    x1, y1, x2, y2 = [int(c) for c in bbox]
    h, w = image.shape[:2]
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    crop = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, crop)


def resize_image(image: np.ndarray, max_size: int = 1200):
    """
    Resize image if larger than max_size.
    
    Args:
        image: Input image array
        max_size: Maximum dimension (width or height)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    return image


def load_image(image_path: str):
    """Load image from file"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img

"""
Food Detection & Classification Package
========================================
A modular package for detecting and classifying food items using YOLOE and MobileCLIP.
"""

__version__ = "1.0.0"
__author__ = "thiennvu1914"

from .core.pipeline import FoodDetectionPipeline
from .core.detector import YOLOEFoodDetector
from .core.embedder import MobileCLIPEmbedder
from .core.classifier import FoodClassifier

__all__ = [
    "FoodDetectionPipeline",
    "YOLOEFoodDetector",
    "MobileCLIPEmbedder",
    "FoodClassifier",
]

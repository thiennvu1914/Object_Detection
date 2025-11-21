"""Core package initialization"""

from .pipeline import FoodDetectionPipeline
from .detector import YOLOEFoodDetector
from .embedder import MobileCLIPEmbedder
from .classifier import FoodClassifier

__all__ = [
    "FoodDetectionPipeline",
    "YOLOEFoodDetector", 
    "MobileCLIPEmbedder",
    "FoodClassifier",
]

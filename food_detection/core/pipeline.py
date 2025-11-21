"""
Food Detection Pipeline Module
===============================
Complete pipeline for food detection and classification.
"""
from pathlib import Path
import time
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

from .detector import YOLOEFoodDetector
from .embedder import MobileCLIPEmbedder
from .classifier import FoodClassifier


class FoodDetectionPipeline:
    """Complete food detection and classification pipeline"""
    
    def __init__(
        self,
        yoloe_model: str = "models/yoloe-11l-seg-pf.pt",
        mobileclip_model: str = "models/mobileclip_s2",
        ref_images_dir: str = "data/ref_images"
    ):
        """
        Initialize pipeline with models.
        
        Args:
            yoloe_model: Path to YOLOE model
            mobileclip_model: Path to MobileCLIP model directory
            ref_images_dir: Path to reference images directory
        """
        self.detector = YOLOEFoodDetector(yoloe_model)
        self.embedder = MobileCLIPEmbedder(mobileclip_model)
        self.ref_images_dir = Path(ref_images_dir)
        
        # Load reference embeddings
        self.reference_embeddings = self._load_reference_embeddings()
        self.classifier = FoodClassifier(self.reference_embeddings)
    
    def _load_reference_embeddings(self) -> Dict[str, np.ndarray]:
        """Load and generate embeddings for reference images"""
        ref_embeddings = {}
        
        if not self.ref_images_dir.exists():
            raise FileNotFoundError(f"Reference images directory not found: {self.ref_images_dir}")
        
        for class_dir in self.ref_images_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue
            
            class_name = class_dir.name
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            if not image_files:
                continue
            
            # Generate embeddings for all reference images
            embeddings = self.embedder.encode_images_batch([str(f) for f in image_files])
            ref_embeddings[class_name] = embeddings
        
        return ref_embeddings
    
    def process_image(
        self,
        image_path: str,
        conf: float = 0.5,
        return_crops: bool = False
    ) -> Dict:
        """
        Process a single image through the full pipeline.
        
        Args:
            image_path: Path to input image
            conf: Detection confidence threshold
            return_crops: If True, include cropped images in results
            
        Returns:
            Dictionary containing detection results:
            - detections: List of detection dicts with bbox, class, score
            - image_shape: Original image dimensions
            - processing_time: Total processing time in seconds
            - crops: Cropped images (if return_crops=True)
        """
        start_time = time.time()
        
        # 1. Detect objects (returns list of dicts with ensemble filtering)
        detections = self.detector.detect(image_path, conf=conf, filter_method="ensemble")
        
        # Read image for cropping
        image = cv2.imread(image_path)
        
        if len(detections) == 0:
            return {
                'detections': [],
                'image_shape': image.shape,
                'processing_time': time.time() - start_time,
                'crops': []
            }
        
        # 2. Crop detected regions and generate embeddings
        crops = []
        embeddings = []
        
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det['bbox']]
            
            # Add padding
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            crop = image[y1:y2, x1:x2]
            crops.append(crop)
            
            # Generate embedding (embed() takes BGR image array)
            embedding = self.embedder.embed(crop)
            embeddings.append(embedding)
        
        # 3. Classify using embeddings
        classifications = self.classifier.classify_batch(np.array(embeddings))
        
        # 4. Update detections with classification results
        final_detections = []
        for i, (det, (class_name, similarity)) in enumerate(zip(detections, classifications)):
            final_detections.append({
                'bbox': det['bbox'],
                'class': class_name,
                'similarity': float(similarity),
                'confidence': float(det['score']),
                'index': i
            })
        
        # 5. Print classification summary
        from collections import Counter
        class_counts = Counter([det['class'] for det in final_detections])
        print(f"\n📦 CLASSIFICATION RESULTS:")
        print(f"   Detected {len(final_detections)} food items:")
        for class_name, count in sorted(class_counts.items()):
            print(f"      • {count}x {class_name}")
        print()
        
        result = {
            'detections': final_detections,
            'image_shape': image.shape,
            'processing_time': time.time() - start_time
        }
        
        if return_crops:
            result['crops'] = crops
        
        return result
    
    def get_available_classes(self) -> List[str]:
        """Get list of available food classes"""
        return self.classifier.classes

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
        
        # 1. Detect objects
        results = self.detector.detect(image_path, conf=conf)
        boxes = results.boxes
        
        # 2. Filter detections
        boxes = self.detector.remove_large_boxes(boxes, results.orig_img.shape)
        boxes = self.detector.filter_containers(boxes)
        boxes = self.detector.remove_overlaps(boxes)
        
        if len(boxes) == 0:
            return {
                'detections': [],
                'image_shape': results.orig_img.shape,
                'processing_time': time.time() - start_time,
                'crops': []
            }
        
        # 3. Crop detected regions
        image = cv2.imread(image_path)
        crops = []
        crop_files = []
        
        for i, box in enumerate(boxes):
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            
            # Add padding
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            crop = image[y1:y2, x1:x2]
            
            # Save temporary crop
            temp_path = f"/tmp/crop_{i}.jpg"
            cv2.imwrite(temp_path, crop)
            crop_files.append(temp_path)
            
            if return_crops:
                crops.append(crop)
        
        # 4. Generate embeddings
        embeddings = self.embedder.encode_images_batch(crop_files)
        
        # 5. Classify
        classifications = self.classifier.classify_batch(embeddings)
        
        # 6. Build results
        detections = []
        for i, (box, (class_name, similarity)) in enumerate(zip(boxes, classifications)):
            coords = box.xyxy[0].cpu().numpy()
            det_conf = float(box.conf[0].cpu().numpy())
            
            detections.append({
                'bbox': coords.tolist(),
                'class': class_name,
                'similarity': similarity,
                'confidence': det_conf,
                'index': i
            })
        
        result = {
            'detections': detections,
            'image_shape': results.orig_img.shape,
            'processing_time': time.time() - start_time
        }
        
        if return_crops:
            result['crops'] = crops
        
        return result
    
    def get_available_classes(self) -> List[str]:
        """Get list of available food classes"""
        return self.classifier.classes

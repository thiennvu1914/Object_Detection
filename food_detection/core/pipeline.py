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
from ..database import DatabaseManager


class FoodDetectionPipeline:
    """Complete food detection and classification pipeline"""
    
    def __init__(
        self,
        yoloe_model: str = "models/yoloe-11l-seg-pf.pt",
        mobileclip_model: str = "models/mobileclip_s2",
        ref_images_dir: str = "data/ref_images",
        db_path: str = "food_detection.db",
        use_cache: bool = True
    ):
        """
        Initialize pipeline with models.
        
        Args:
            yoloe_model: Path to YOLOE model
            mobileclip_model: Path to MobileCLIP model directory
            ref_images_dir: Path to reference images directory
            db_path: Path to SQLite database
            use_cache: If True, load embeddings from database cache
        """
        self.detector = YOLOEFoodDetector(yoloe_model)
        self.embedder = MobileCLIPEmbedder(mobileclip_model)
        self.ref_images_dir = Path(ref_images_dir)
        self.db = DatabaseManager(db_path)
        self.use_cache = use_cache
        
        # Load reference embeddings (from cache or compute)
        self.reference_embeddings = self._load_reference_embeddings()
        self.classifier = FoodClassifier(self.reference_embeddings)
    
    def _load_reference_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load reference embeddings from database cache or compute from images.
        
        Returns:
            Dictionary mapping class_name to embeddings array
        """
        # Try to load from database cache
        if self.use_cache:
            print("Loading reference embeddings from database cache...")
            cached_embeddings = self.db.load_reference_embeddings()
            
            if cached_embeddings:
                counts = self.db.get_embeddings_count()
                print(f"✓ Loaded cached embeddings: {counts}")
                return cached_embeddings
            else:
                print("  No cached embeddings found, computing from images...")
        
        # Compute embeddings from reference images
        if not self.ref_images_dir.exists():
            raise FileNotFoundError(f"Reference images directory not found: {self.ref_images_dir}")
        
        print(f"Computing embeddings from {self.ref_images_dir}...")
        ref_embeddings = {}
        
        for class_dir in self.ref_images_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue
            
            class_name = class_dir.name
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            if not image_files:
                continue
            
            # Generate embeddings for all reference images
            print(f"  - {class_name}: {len(image_files)} images")
            embeddings = self.embedder.encode_images_batch([str(f) for f in image_files])
            ref_embeddings[class_name] = embeddings
            
            # Cache to database
            if self.use_cache:
                for img_path, embedding in zip(image_files, embeddings):
                    self.db.save_reference_embeddings(class_name, str(img_path), embedding)
        
        if self.use_cache:
            print(f"✓ Cached {sum(len(embs) for embs in ref_embeddings.values())} embeddings to database")
        
        return ref_embeddings
    
    def process_image(
        self,
        image_path: str,
        conf: float = 0.25,
        filter_method: str = "ensemble",
        return_crops: bool = False,
        save_to_db: bool = True
    ) -> Dict:
        """
        Process a single image through the full pipeline.
        
        Args:
            image_path: Path to input image
            conf: Detection confidence threshold
            filter_method: Filtering method ('ensemble', 'spatial', 'size', 'ml', 'none')
            return_crops: If True, include cropped images in results
            save_to_db: If True, save results to database
            
        Returns:
            Dictionary containing detection results:
            - detections: List of detection dicts with bbox, class, score
            - image_shape: Original image dimensions
            - processing_time: Total processing time in seconds
            - crops: Cropped images (if return_crops=True)
        """
        start_time = time.time()
        
        # 1. Detect objects (returns list of dicts with ensemble filtering)
        detections = self.detector.detect(image_path, conf=conf, filter_method=filter_method)
        
        # Read image for cropping
        image = cv2.imread(image_path)
        
        if len(detections) == 0:
            # Save empty session if requested
            if save_to_db:
                filename = Path(image_path).name
                self.db.save_detection_session(filename, [])
                
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
        # Use a higher threshold (0.55) to filter out non-food items like watches/phones
        # Real food items usually have similarity > 0.8
        classifications = self.classifier.classify_batch(np.array(embeddings), threshold=0.55)
        
        # 4. Update detections with classification results
        final_detections = []
        for i, (det, (class_name, similarity)) in enumerate(zip(detections, classifications)):
            # Skip unknown items (low similarity)
            if class_name == "unknown":
                continue
                
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
        print(f"\nCLASSIFICATION RESULTS:")
        print(f"   Detected {len(final_detections)} food items:")
        for class_name, count in sorted(class_counts.items()):
            print(f"      {count}x {class_name}")
        print()
        
        result = {
            'detections': final_detections,
            'image_shape': image.shape,
            'processing_time': time.time() - start_time
        }
        
        if return_crops:
            result['crops'] = crops
            
        # Save to database
        if save_to_db:
            try:
                filename = Path(image_path).name
                self.db.save_detection_session(filename, final_detections)
                print(f"✓ Saved results to database (session for {filename})")
            except Exception as e:
                print(f"⚠️ Failed to save to database: {e}")
        
        return result
    
    def get_available_classes(self) -> List[str]:
        """Get list of available food classes"""
        return self.classifier.classes

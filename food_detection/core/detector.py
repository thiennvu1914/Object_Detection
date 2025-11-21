"""
YOLOE Food Detector Module
===========================
Smart detection with automatic filtering.
"""
from pathlib import Path
import numpy as np
from ultralytics import YOLOE


class YOLOEFoodDetector:
    """YOLOE-based food detector with smart filtering"""
    
    def __init__(self, model_path: str = "models/yoloe-11l-seg-pf.pt"):
        """
        Initialize YOLOE detector.
        
        Args:
            model_path: Path to YOLOE model file
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = YOLOE(str(model_path))
        
    def detect(self, image_path: str, conf: float = 0.25, iou: float = 0.5):
        """
        Detect objects in image.
        
        Args:
            image_path: Path to input image
            conf: Confidence threshold
            iou: IoU threshold for NMS
            
        Returns:
            Detection results from YOLOE
        """
        results = self.model.predict(image_path, conf=conf, iou=iou, verbose=False)
        return results[0]
    
    def filter_containers(self, boxes):
        """Remove background/container objects (table, tray, board, etc.)"""
        container_classes = {
            'table', 'board', 'dining table', 'desk', 'bar stool',
            'chair', 'couch', 'bed', 'bench', 'plank', 'wood',
            'surfboard', 'skateboard', 'cutting board', 'shelf',
            'cigar box', 'box', 'container', 'tray', 'plaque',
            'bottle cap', 'lid', 'plate', 'saucer', 'linen',
            'napkin', 'cloth', 'towel', 'mat', 'tablecloth',
            'close-up', 'beige', 'background', 'surface', 'texture'
        }
        
        filtered = []
        for b in boxes:
            cls_name = self.model.names[int(b.cls[0].item())].lower()
            if cls_name not in container_classes:
                filtered.append(b)
        
        return filtered
    
    def remove_overlaps(self, boxes, threshold: float = 0.95):
        """Remove boxes with high overlap (>= threshold)"""
        if len(boxes) <= 1:
            return boxes
        
        # Sort by area (largest first)
        sorted_boxes = sorted(boxes, key=lambda b: (
            (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
        ), reverse=True)
        
        keep = []
        for i, box in enumerate(sorted_boxes):
            should_keep = True
            for kept_box in keep:
                overlap = self._calculate_overlap_ratio(
                    box.xyxy[0].cpu().numpy(),
                    kept_box.xyxy[0].cpu().numpy()
                )
                if overlap >= threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(box)
        
        return keep
    
    def remove_large_boxes(self, boxes, image_shape, threshold: float = 0.7):
        """Remove boxes covering more than threshold% of image"""
        if len(boxes) == 0:
            return boxes
        
        h, w = image_shape[:2]
        image_area = h * w
        
        filtered = []
        for box in boxes:
            coords = box.xyxy[0].cpu().numpy()
            box_area = (coords[2] - coords[0]) * (coords[3] - coords[1])
            ratio = box_area / image_area
            
            if ratio <= threshold:
                filtered.append(box)
        
        return filtered
    
    @staticmethod
    def _calculate_overlap_ratio(small_box, large_box):
        """Calculate overlap ratio of small_box inside large_box"""
        x1_min, y1_min, x1_max, y1_max = small_box
        x2_min, y2_min, x2_max, y2_max = large_box
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        small_area = (x1_max - x1_min) * (y1_max - y1_min)
        
        return inter_area / small_area if small_area > 0 else 0.0

"""
YOLOE Food Detector Module
===========================
Smart filtering for food detection with ensemble methods.

4 Filtering Methods:
1. SPATIAL FILTERING - Find food region clusters
2. SIZE-BASED FILTERING - Remove size outliers
3. ML CLASSIFIER - Score based on features
4. ENSEMBLE - Voting from 3 methods (RECOMMENDED)
"""
from ultralytics import YOLOE
import cv2
import numpy as np
from pathlib import Path
import torch
from typing import List, Optional


class YOLOEFoodDetector:
    """Smart filtering for food items - no keywords needed"""
    
    def __init__(self, model_path="models/yoloe-11l-seg-pf.pt"):
        print("Loading YOLOE model...")
        self.model = YOLOE(model_path)
        print("✓ Model loaded")
    
    def detect(self, image_path, conf=0.5, filter_method="ensemble"):
        """
        Detect food items with smart filtering.
        
        Args:
            image_path: Path to image
            conf: Confidence threshold (default: 0.5)
            filter_method: 'ensemble', 'spatial', 'size', 'ml', or 'none'
        
        Returns:
            List of detection dicts with 'bbox', 'label', 'score'
        """
        # Get raw predictions
        result = self.predict(image_path, conf=conf)
        
        # Apply filtering (returns list of box objects)
        if filter_method == "ensemble":
            filtered_boxes = self.ensemble_filter(result)
        elif filter_method == "spatial":
            filtered_boxes = self.spatial_filter(result)
        elif filter_method == "size":
            filtered_boxes = self.size_filter(result)
        elif filter_method == "ml":
            filtered_boxes = self.ml_filter(result)
        else:
            # No filtering
            filtered_boxes = list(result.boxes) if result.boxes is not None else []
        
        # Convert box objects to dicts
        detections = []
        for box in filtered_boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            cls_id = int(box.cls[0].cpu().item())
            label = self.model.names[cls_id]
            
            detections.append({
                'bbox': xyxy.tolist(),
                'label': label,
                'score': float(conf)
            })
        
        return detections
    
    def predict(self, image_path, conf=0.25, iou=0.5):
        """Run YOLOE prediction"""
        results = self.model.predict(image_path, conf=conf, iou=iou, verbose=False)
        return results[0]
    
    # ==================== POST-PROCESSING ====================
    def calculate_iou(self, box1, box2):
        """Tính IoU (Intersection over Union) giữa 2 boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def remove_container(self, boxes):
        """
        XÁC ĐỊNH VÀ LOẠI BỎ CONTAINER (khay/bàn/board/table):
        - LOẠI BỎ box với class = table, board, dining table, desk
        - Không dựa vào size
        """
        if len(boxes) == 0:
            return boxes

        # Container classes cần loại bỏ (không phải food)
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
        removed = []
        
        for b in boxes:
            cls_name = self.model.names[int(b.cls[0].item())].lower()
            
            if cls_name in container_classes:
                removed.append(cls_name)
            else:
                filtered.append(b)
        
        if removed:
            print(f"  → Removed containers: {', '.join(set(removed))}")
        
        return filtered
    
    def calculate_overlap_ratio(self, small_box, large_box):
        """
        Tính % diện tích của small_box nằm trong large_box
        Return: 0.0 - 1.0 (1.0 = hoàn toàn nằm trong)
        """
        x1_min, y1_min, x1_max, y1_max = small_box
        x2_min, y2_min, x2_max, y2_max = large_box
        
        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h
        
        # Area of small box
        small_area = (x1_max - x1_min) * (y1_max - y1_min)
        
        if small_area == 0:
            return 0.0
        
        # % diện tích small_box nằm trong large_box
        return inter_area / small_area
    
    def remove_inner(self, boxes, overlap_threshold=0.95):
        """
        RULE: Loại bỏ khung nhỏ nằm hoàn toàn/trùng >= 95% trong khung lớn
        
        Ví dụ:
        - BISCUIT (nhỏ) nằm 95% trong ECLAIR (lớn) → xóa BISCUIT
        - BAGEL (nhỏ) nằm 92% trong DONUT (lớn) → xóa BAGEL
        """
        if len(boxes) <= 1:
            return boxes

        # Sort by area descending
        info = []
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            cls_name = self.model.names[int(b.cls[0].item())]
            info.append({'box': b, 'xyxy': xyxy, 'area': area, 'name': cls_name, 'keep': True})
        
        info.sort(key=lambda x: x['area'], reverse=True)
        
        removed_parts = []
        
        # So sánh từng cặp (lớn với nhỏ)
        for i in range(len(info)):
            if not info[i]['keep']:
                continue
            
            for j in range(i + 1, len(info)):
                if not info[j]['keep']:
                    continue
                
                larger_box = info[i]['xyxy']
                smaller_box = info[j]['xyxy']
                
                # Tính % diện tích smaller_box nằm trong larger_box
                overlap_ratio = self.calculate_overlap_ratio(smaller_box, larger_box)
                
                # Nếu khung nhỏ nằm >= 90-95% trong khung lớn → xóa khung nhỏ
                if overlap_ratio >= overlap_threshold:
                    removed_parts.append(f"{info[j]['name']} (inside {info[i]['name']}, {overlap_ratio*100:.0f}%)")
                    info[j]['keep'] = False
        
        if removed_parts:
            print(f"  → Removed inner boxes: {', '.join(removed_parts)}")
        
        return [x['box'] for x in info if x['keep']]
    def normalize_conf_area(self, boxes):
        """
        RULE: Loại bỏ outliers (confidence thấp hoặc size bất thường)
        - Conf < 50% median → low confidence, bỏ
        - Area < 20% median hoặc > 400% median → outlier, bỏ (more lenient)
        - Box quá lớn (>70% ảnh) → có thể là background, bỏ
        """
        if len(boxes) <= 2:
            return boxes

        confs = [b.conf[0].item() for b in boxes]
        areas = []
        
        # Lấy kích thước ảnh từ box đầu tiên
        img_area = None
        for b in boxes:
            x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
            if img_area is None:
                # Ước tính kích thước ảnh (giả sử box không vượt quá ảnh)
                img_area = max(x2, 640) * max(y2, 640)
            areas.append((x2-x1)*(y2-y1))

        median_conf = np.median(confs)
        median_area = np.median(areas)

        final = []
        removed_outliers = []
        
        for b in boxes:
            conf = b.conf[0].item()
            x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
            width = x2 - x1
            height = y2 - y1
            area = width * height
            cls_name = self.model.names[int(b.cls[0].item())]

            # Check if box is too large (>70% of estimated image area)
            if img_area and area > 0.7 * img_area:
                removed_outliers.append(f"{cls_name} (too large={area:.0f}, covers {100*area/img_area:.0f}% of image)")
                continue

            # Check confidence (more lenient)
            if conf < 0.5 * median_conf:
                removed_outliers.append(f"{cls_name} (low conf={conf:.2f})")
                continue
            
            # Check size outlier (more lenient range)
            if not (0.2 * median_area <= area <= 4.0 * median_area):
                removed_outliers.append(f"{cls_name} (outlier size={area:.0f})")
                continue

            final.append(b)
        
        if removed_outliers:
            print(f"  → Removed outliers: {', '.join(removed_outliers)}")

        return final
    def remove_too_large_boxes(self, boxes, threshold=0.7):
        """
        Loại bỏ boxes quá lớn (có thể là background/toàn bộ ảnh)
        threshold: Box chiếm >70% ảnh sẽ bị loại
        """
        if len(boxes) == 0:
            return boxes
        
        # Ước tính kích thước ảnh
        max_x, max_y = 0, 0
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
        
        img_area = max_x * max_y
        
        final = []
        removed_large = []
        
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            box_area = (x2 - x1) * (y2 - y1)
            ratio = box_area / img_area if img_area > 0 else 0
            cls_name = self.model.names[int(b.cls[0].item())]
            
            if ratio > threshold:
                removed_large.append(f"{cls_name} (covers {100*ratio:.0f}% of image)")
            else:
                final.append(b)
        
        if removed_large:
            print(f"  → Removed too-large boxes: {', '.join(removed_large)}")
        
        return final
    
    def post_process(self, boxes):
        """
        Pipeline xử lý 4 bước:
        0. Loại bỏ boxes quá lớn (có thể là toàn bộ ảnh)
        1. Xóa background (khay/bàn lớn) - CHỈ CHẠY 1 LẦN
        2. Xóa parts (khung nhỏ nằm trong object)
        3. Xóa outliers (conf thấp, size bất thường)
        """
        initial_count = len(boxes)
        
        print(f"\n  🔧 Post-processing: {initial_count} boxes")
        
        # Step 0: Remove too-large boxes first
        boxes = self.remove_too_large_boxes(boxes, threshold=0.7)
        
        # Step 1: Remove container CHỈ 1 LẦN
        boxes = self.remove_container(boxes)
        
        # Step 2: Remove inner boxes (duplicates)
        boxes = self.remove_inner(boxes)
        
        # Step 3: Remove outliers
        boxes = self.normalize_conf_area(boxes)
        
        final_count = len(boxes)
        print(f"  ✓ Final: {final_count} food items ({initial_count - final_count} removed)\n")
        
        return boxes
    
    # ==================== METHOD 1: SPATIAL FILTERING ====================
    def spatial_filter(self, result):
        """
        Tìm vùng tập trung objects (food region) bằng clustering
        Loại bỏ objects xa center cluster
        """
        boxes = result.boxes
        if len(boxes) == 0:
            return []
        
        # Tính center của mỗi box
        centers = []
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            centers.append([cx, cy])
        
        centers = np.array(centers)
        
        # Tìm center của cluster (median để robust với outliers)
        cluster_center = np.median(centers, axis=0)
        
        # Tính khoảng cách từ mỗi box đến cluster center
        distances = np.sqrt(np.sum((centers - cluster_center)**2, axis=1))
        
        # Threshold = median + 1.5 * MAD (Median Absolute Deviation)
        mad = np.median(np.abs(distances - np.median(distances)))
        threshold = np.median(distances) + 1.5 * mad
        
        # Giữ boxes gần cluster center
        food_indices = np.where(distances <= threshold)[0]
        filtered_boxes = [boxes[i] for i in food_indices]
        
        # Post-processing: Remove containers & merge duplicates
        filtered_boxes = self.post_process(filtered_boxes)
        
        return filtered_boxes
    
    # ==================== METHOD 2: SIZE-BASED FILTERING ====================
    def size_filter(self, result):
        """
        Clustering theo size (area)
        Loại outliers (quá lớn = background, quá nhỏ = noise)
        """
        boxes = result.boxes
        if len(boxes) == 0:
            return []
        
        # Tính area của mỗi box
        areas = []
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            areas.append(w * h)
        
        areas = np.array(areas)
        
        # Loại outliers bằng IQR method
        q1 = np.percentile(areas, 25)
        q3 = np.percentile(areas, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Giữ boxes trong range hợp lý
        food_indices = np.where((areas >= lower_bound) & (areas <= upper_bound))[0]
        filtered_boxes = [boxes[i] for i in food_indices]
        
        # Post-processing: Remove containers & merge duplicates
        filtered_boxes = self.post_process(filtered_boxes)
        
        return filtered_boxes
    
    # ==================== METHOD 3: ML CLASSIFIER ====================
    def ml_filter(self, result):
        """
        Score dựa trên multiple features:
        - Area (normalized)
        - Aspect ratio
        - Position (center region gets higher score)
        - Confidence
        
        Không cần training data!
        """
        boxes = result.boxes
        if len(boxes) == 0:
            return []
        
        # Get image dimensions
        img_h, img_w = result.orig_shape
        
        scores = []
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].item()
            
            # Feature 1: Normalized area (0-1)
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            area = w * h
            normalized_area = area / (img_w * img_h)
            
            # Feature 2: Aspect ratio (prefer square-ish objects)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            aspect_score = 1.0 / (1.0 + abs(aspect_ratio - 1.5))  # Peak at 1.5
            
            # Feature 3: Center position (prefer center region)
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            center_x = cx / img_w
            center_y = cy / img_h
            
            # Distance from image center (0.5, 0.5)
            center_dist = np.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
            position_score = 1.0 - center_dist  # Max at center
            
            # Feature 4: Confidence
            conf_score = conf
            
            # Weighted score
            final_score = (
                0.2 * (normalized_area * 20) +  # Prefer medium size
                0.2 * aspect_score +
                0.3 * position_score +
                0.3 * conf_score
            )
            
            scores.append(final_score)
        
        scores = np.array(scores)
        
        # Threshold = mean - 0.5 * std (keep above average)
        threshold = np.mean(scores) - 0.5 * np.std(scores)
        
        food_indices = np.where(scores >= threshold)[0]
        filtered_boxes = [boxes[i] for i in food_indices]
        
        # Post-processing: Remove containers & merge duplicates
        filtered_boxes = self.post_process(filtered_boxes)
        
        return filtered_boxes
    
    # ==================== METHOD 4: ENSEMBLE ====================
    def ensemble_filter(self, result):
        """
        Voting từ 3 methods
        Box được giữ nếu >= 2/3 methods đồng ý
        """
        boxes = result.boxes
        if len(boxes) == 0:
            return []
        
        print(f"\n🔄 Running 3 filtering methods...")
        
        # Get results from 3 methods WITHOUT post-processing
        # (chỉ remove inner + outliers, KHÔNG remove container)
        spatial_boxes = self.spatial_filter_no_container(result)
        size_boxes = self.size_filter_no_container(result)
        ml_boxes = self.ml_filter_no_container(result)
        
        print(f"  - Spatial: {len(spatial_boxes)} items")
        print(f"  - Size: {len(size_boxes)} items")
        print(f"  - ML: {len(ml_boxes)} items")
        
        # Convert to sets of indices (use array index instead of id)
        spatial_indices = set()
        for i, box in enumerate(boxes):
            if any(torch.equal(box.xyxy, sb.xyxy) for sb in spatial_boxes):
                spatial_indices.add(i)
        
        size_indices = set()
        for i, box in enumerate(boxes):
            if any(torch.equal(box.xyxy, sb.xyxy) for sb in size_boxes):
                size_indices.add(i)
        
        ml_indices = set()
        for i, box in enumerate(boxes):
            if any(torch.equal(box.xyxy, mb.xyxy) for mb in ml_boxes):
                ml_indices.add(i)
        
        # Voting: keep if >= 1 method agrees (lenient for small datasets)
        food_boxes = []
        for i, box in enumerate(boxes):
            votes = sum([
                i in spatial_indices,
                i in size_indices,
                i in ml_indices
            ])
            
            if votes >= 1:
                food_boxes.append(box)
        
        print(f"  - After voting (>=1 vote): {len(food_boxes)} items")
        
        # Final post-processing: Remove too-large, container, and inner boxes
        food_boxes = self.remove_too_large_boxes(food_boxes, threshold=0.7)
        food_boxes = self.remove_container(food_boxes)
        food_boxes = self.remove_inner(food_boxes)
        
        print(f"  ✓ Final result: {len(food_boxes)} food items\n")
        
        return food_boxes
    
    def spatial_filter_no_container(self, result):
        """Spatial filter WITHOUT removing container"""
        boxes = result.boxes
        if len(boxes) == 0:
            return []
        
        centers = []
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            centers.append([cx, cy])
        
        centers = np.array(centers)
        cluster_center = np.median(centers, axis=0)
        distances = np.sqrt(np.sum((centers - cluster_center)**2, axis=1))
        mad = np.median(np.abs(distances - np.median(distances)))
        threshold = np.median(distances) + 1.5 * mad
        food_indices = np.where(distances <= threshold)[0]
        return [boxes[i] for i in food_indices]
    
    def size_filter_no_container(self, result):
        """Size filter WITHOUT removing container"""
        boxes = result.boxes
        if len(boxes) == 0:
            return []
        
        areas = []
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            areas.append(w * h)
        
        areas = np.array(areas)
        q1 = np.percentile(areas, 25)
        q3 = np.percentile(areas, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        food_indices = np.where((areas >= lower_bound) & (areas <= upper_bound))[0]
        return [boxes[i] for i in food_indices]
    
    def ml_filter_no_container(self, result):
        """ML filter WITHOUT removing container"""
        boxes = result.boxes
        if len(boxes) == 0:
            return []
        
        img_h, img_w = result.orig_shape
        scores = []
        
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].item()
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            area = w * h
            normalized_area = area / (img_w * img_h)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            aspect_score = 1.0 / (1.0 + abs(aspect_ratio - 1.5))
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            center_x = cx / img_w
            center_y = cy / img_h
            center_dist = np.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
            position_score = 1.0 - center_dist
            conf_score = conf
            final_score = (
                0.2 * (normalized_area * 20) +
                0.2 * aspect_score +
                0.3 * position_score +
                0.3 * conf_score
            )
            scores.append(final_score)
        
        scores = np.array(scores)
        threshold = np.mean(scores) - 0.5 * np.std(scores)
        food_indices = np.where(scores >= threshold)[0]
        return [boxes[i] for i in food_indices]
    
    # ==================== VISUALIZATION ====================
    def visualize_results(self, image_path, food_boxes, method_name, save_path=None, rename_to_obj=True):
        """Draw bounding boxes on image"""
        img = cv2.imread(str(image_path))
        
        for box in food_boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            class_name = self.model.names[cls]
            
            # Draw box
            color = (0, 255, 0)  # Green
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            
            # Draw label - đổi thành "obj" nếu rename_to_obj=True
            if rename_to_obj:
                label = f"obj {conf:.2f}"
            else:
                label = f"{class_name} {conf:.2f}"
            
            cv2.putText(img, label, (xyxy[0], xyxy[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add method name
        cv2.putText(img, f"Method: {method_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if save_path:
            cv2.imwrite(str(save_path), img)
            print(f"✓ Saved to: {save_path}")
        
        return img
    
    # ==================== COMPARE ALL METHODS ====================
    def compare_all(self, image_path):
        """So sánh tất cả methods"""
        print("\n" + "="*60)
        print(f"COMPARING ALL METHODS: {Path(image_path).name}")
        print("="*60)
        
        # Run prediction
        result = self.predict(image_path)
        total_boxes = len(result.boxes)
        
        print(f"\nTotal detected: {total_boxes} objects\n")
        
        # Method 1: Spatial
        spatial_boxes = self.spatial_filter(result)
        print(f"1. SPATIAL FILTERING: {len(spatial_boxes)} food items")
        
        # Method 2: Size
        size_boxes = self.size_filter(result)
        print(f"2. SIZE-BASED FILTERING: {len(size_boxes)} food items")
        
        # Method 3: ML
        ml_boxes = self.ml_filter(result)
        print(f"3. ML CLASSIFIER: {len(ml_boxes)} food items")
        
        # Method 4: Ensemble
        ensemble_boxes = self.ensemble_filter(result)
        print(f"4. ENSEMBLE (Voting): {len(ensemble_boxes)} food items ⭐")
        
        # Save all results
        output_dir = Path("outputs/comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        img_name = Path(image_path).stem
        
        self.visualize_results(image_path, spatial_boxes, "Spatial", 
                              output_dir / f"{img_name}_spatial.jpg")
        self.visualize_results(image_path, size_boxes, "Size-Based", 
                              output_dir / f"{img_name}_size.jpg")
        self.visualize_results(image_path, ml_boxes, "ML Classifier", 
                              output_dir / f"{img_name}_ml.jpg")
        self.visualize_results(image_path, ensemble_boxes, "Ensemble", 
                              output_dir / f"{img_name}_ensemble.jpg")
        
        print(f"\n✓ All results saved to: {output_dir}")
        print("="*60)
        
        return {
            'spatial': spatial_boxes,
            'size': size_boxes,
            'ml': ml_boxes,
            'ensemble': ensemble_boxes
        }


def main():
    """CLI interface"""
    import sys
    if len(sys.argv) < 3:
        print("""
YOLOE Smart Filtering - No Keywords!
=====================================

Usage:
  python yoloe_food.py <method> <image_path>

Methods:
  spatial   - Spatial clustering (loại objects xa food region)
  size      - Size-based filtering (loại outliers)
  ml        - ML classifier (score dựa trên features)
  ensemble  - Ensemble voting (kết hợp 3 methods) ⭐ RECOMMENDED
  compare   - So sánh tất cả methods

Examples:
  python yoloe_food.py ensemble data/images/image_01.jpg
  python yoloe_food.py compare data/images/image_01.jpg
        """)
        return
    
    method = sys.argv[1]
    image_path = sys.argv[2]
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Initialize detector
    filter = YOLOEFoodDetector()
    
    if method == "compare":
        filter.compare_all(image_path)
    else:
        # Run single method
        result = filter.predict(image_path)
        
        method_map = {
            'spatial': ('Spatial Filtering', filter.spatial_filter),
            'size': ('Size-Based Filtering', filter.size_filter),
            'ml': ('ML Classifier', filter.ml_filter),
            'ensemble': ('Ensemble Voting', filter.ensemble_filter)
        }
        
        if method not in method_map:
            print(f"❌ Unknown method: {method}")
            return
        
        method_name, method_func = method_map[method]
        
        print(f"\nRunning {method_name}...")
        
        # Show all detections first
        print(f"\n📦 ALL DETECTIONS ({len(result.boxes)} objects):")
        for i, box in enumerate(result.boxes):
            cls = int(box.cls[0].item())
            conf = box.conf[0].item()
            class_name = filter.model.names[cls]
            print(f"  {i+1}. {class_name} (Conf: {conf:.3f})")
        
        food_boxes = method_func(result)
        
        print(f"\n✓ Total detected: {len(result.boxes)} objects")
        print(f"✓ Food items: {len(food_boxes)} objects")
        
        # Print details
        print(f"\n📋 DETECTED FOOD ITEMS:")
        for i, box in enumerate(food_boxes):
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            class_name = filter.model.names[cls]
            xyxy = box.xyxy[0].cpu().numpy()
            
            print(f"  {i+1}. obj (Original: {class_name}, Conf: {conf:.3f})")
        
        # Save result với nhãn "obj"
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        img_name = Path(image_path).stem
        save_path = output_dir / f"{img_name}_{method}.jpg"
        
        filter.visualize_results(image_path, food_boxes, method_name, save_path, rename_to_obj=True)


if __name__ == "__main__":
    main()

import onnxruntime as ort
import numpy as np
import cv2

class YOLO11Detector:
    def __init__(self, model_path="models/yolov11_food.onnx", input_size=None):
        """
        Args:
            model_path: Đường dẫn model ONNX
            input_size: Size ảnh input (nếu None thì lấy từ model)
        """
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        
        # Lấy input shape từ model (ONNX export thường fix size)
        input_shape = self.session.get_inputs()[0].shape
        if input_size is None and len(input_shape) == 4:
            self.input_size = input_shape[2]  # Sử dụng size từ model
            print(f"📐 Auto-detected input size from model: {self.input_size}x{self.input_size}")
        else:
            self.input_size = input_size if input_size else 640
            print(f"📐 Using specified input size: {self.input_size}x{self.input_size}")

    def preprocess(self, img):
        """Preprocessing với letterbox để giữ aspect ratio"""
        h0, w0 = img.shape[:2]
        
        # Letterbox resize - giữ nguyên tỷ lệ, padding phần còn thiếu
        r = self.input_size / max(h0, w0)
        new_w = int(w0 * r)
        new_h = int(h0 * r)
        
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Padding
        pad_w = self.input_size - new_w
        pad_h = self.input_size - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Normalize
        img_padded = img_padded[:, :, ::-1] / 255.0
        img_tensor = np.transpose(img_padded, (2, 0, 1))[None].astype(np.float32)
        
        return img_tensor, (w0, h0, r, left, top)

    def scale_box(self, box, orig_w, orig_h, ratio, pad_x, pad_y):
        """Scale box về kích thước ảnh gốc với letterbox"""
        x1, y1, x2, y2 = box
        
        # Remove padding
        x1 = (x1 - pad_x) / ratio
        y1 = (y1 - pad_y) / ratio
        x2 = (x2 - pad_x) / ratio
        y2 = (y2 - pad_y) / ratio
        
        # Clip to image bounds
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        
        return [int(x1), int(y1), int(x2), int(y2)]

    def nms(self, boxes, scores, iou_thresh=0.45):
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []
            
        idxs = np.argsort(scores)[::-1]
        keep = []

        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)

            if len(idxs) == 1:
                break

            ious = self.iou(boxes[i], boxes[idxs[1:]])
            idxs = idxs[1:][ious < iou_thresh]

        return keep

    def iou(self, box, boxes):
        """Calculate IoU between one box and array of boxes"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return inter / (area_box + area_boxes - inter + 1e-6)

    def predict(self, img, conf_thres=0.05, iou_thres=0.45, min_box_size=5):
        """
        Predict với nhiều cải tiến cho vật nhỏ
        
        Args:
            img: Input image
            conf_thres: Confidence threshold (giảm xuống 0.1 cho vật nhỏ)
            iou_thres: NMS IoU threshold
            min_box_size: Kích thước tối thiểu của box (pixel)
        """
        tensor, (orig_w, orig_h, ratio, pad_x, pad_y) = self.preprocess(img)
        preds = self.session.run(None, {self.input_name: tensor})[0]

        # Handle different output formats
        # Shape can be: (1, 84, 8400), (1, 8400, 84), or (1, 5, 8400) for single-class models
        if len(preds.shape) == 3:
            batch_size, dim1, dim2 = preds.shape
            
            # Check which dimension is the feature dimension (4 + num_classes)
            if dim1 > dim2:  # (1, 8400, 84) or (1, 8400, 5)
                preds = preds[0]  # (8400, features)
            else:  # (1, 84, 8400) or (1, 5, 8400)
                preds = preds.transpose(0, 2, 1)[0]  # (8400, features)
        elif len(preds.shape) == 2:
            # Already in correct shape (8400, features)
            pass
        else:
            raise ValueError(f"Unexpected output shape: {preds.shape}")
        
        # Validate output has at least 5 dimensions (x, y, w, h, class_score)
        if preds.shape[1] < 5:
            raise ValueError(f"Output must have at least 5 features, got {preds.shape[1]}")

        boxes = []
        scores = []
        class_ids = []

        boxes = []
        scores = []
        class_ids = []

        for row in preds:
            x_center, y_center, width, height = row[:4]
            
            # Handle both single-class and multi-class models
            if preds.shape[1] == 5:  # Single class: [x, y, w, h, objectness]
                max_score = row[4]
                class_id = 0  # Only one class
            else:  # Multi-class: [x, y, w, h, class1_score, class2_score, ...]
                class_scores = row[4:]
                max_score = np.max(class_scores)
                class_id = np.argmax(class_scores)

            if max_score < conf_thres:
                continue

            # Convert to corner format
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            # Skip invalid boxes
            if width <= 0 or height <= 0:
                continue

            scaled = self.scale_box([x1, y1, x2, y2], orig_w, orig_h, ratio, pad_x, pad_y)
            
            # Filter box không hợp lệ
            if scaled[0] >= scaled[2] or scaled[1] >= scaled[3]:
                continue
                
            boxes.append(scaled)
            scores.append(float(max_score))
            class_ids.append(int(class_id))

        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)

        # Apply NMS
        keep_idx = self.nms(boxes, scores, iou_thresh=iou_thres)

        final_boxes = []
        for i in keep_idx:
            final_boxes.append({
                'box': boxes[i].tolist(),
                'score': float(scores[i]),
                'class_id': int(class_ids[i])
            })
        
        return final_boxes


# Usage example with visualization
def visualize_detections(img, detections, class_names=None):
    """
    Vẽ bounding boxes lên ảnh
    
    Args:
        img: Input image
        detections: List of detection dicts from predict()
        class_names: List of class names (optional)
    """
    img_vis = img.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        score = det['score']
        class_id = det['class_id']
        
        # Draw box
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"Class {class_id}: {score:.2f}"
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}: {score:.2f}"
        
        cv2.putText(img_vis, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_vis


# Test với nhiều confidence thresholds
def test_multiple_thresholds(detector, img):
    """Test với nhiều threshold để tìm giá trị tốt nhất"""
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    for conf in thresholds:
        detections = detector.predict(img, conf_thres=conf)
        print(f"Confidence {conf}: Found {len(detections)} objects")
        
        for i, det in enumerate(detections):
            box = det['box']
            w = box[2] - box[0]
            h = box[3] - box[1]
            print(f"  Object {i+1}: score={det['score']:.3f}, size={w}x{h}px")
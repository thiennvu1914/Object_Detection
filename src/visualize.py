import cv2
import time
import numpy as np

from detect import YOLO11Detector
from crop_food import crop_food
from embed import MobileCLIP2Embedder
from match import find_best_match

# Global variables (will be loaded when needed)
db_vectors = None
db_labels = None
detector = None
embedder = None

def _load_models():
    """Lazy load models only when needed"""
    global db_vectors, db_labels, detector, embedder
    
    if detector is None:
        # Load DB
        db_vectors = np.load("db/vectors.npy")
        db_labels = open("db/labels.txt").read().splitlines()
        
        # Load models
        detector = YOLO11Detector("models/yolo11_food_trained.onnx")
        embedder = MobileCLIP2Embedder("models/mobileclip_s2/mobileclip_s2.pt")
    
    return detector, embedder, db_vectors, db_labels

# Màu sắc cố định cho các class ban đầu (BGR format)
BASE_CLASS_COLORS = {
    'coconut': (255, 191, 0),    # Deep Sky Blue
    'cua': (0, 165, 255),        # Orange
    'macaron': (203, 192, 255),  # Pink
    'meden': (0, 255, 255),      # Yellow
    'melon': (0, 255, 0),        # Green
    'unknown': (128, 128, 128)   # Gray
}

# Dynamic color cache cho các class mới
_dynamic_colors = {}

def get_class_color(class_name, seed=42):
    """
    Lấy màu cho class:
    - Nếu là class ban đầu (coconut, cua, macaron, meden, melon) -> dùng màu cố định
    - Nếu là class mới -> tạo màu random dựa trên hash của tên class
    """
    if class_name in BASE_CLASS_COLORS:
        return BASE_CLASS_COLORS[class_name]
    
    # Tạo màu mới cho class chưa có
    if class_name not in _dynamic_colors:
        # Sử dụng hash của class name để tạo màu nhất quán
        hash_value = hash(class_name)
        np.random.seed(abs(hash_value) % (2**32))
        color = tuple(int(c) for c in np.random.randint(50, 255, 3))
        _dynamic_colors[class_name] = color
    
    return _dynamic_colors[class_name]


def generate_colors(num_colors, seed=42):
    """Generate random distinct colors"""
    np.random.seed(seed)
    colors = []
    for i in range(num_colors):
        color = tuple(int(c) for c in np.random.randint(50, 255, 3))
        colors.append(color)
    return colors


def draw_box(img, box, label, score, color=None):
    """
    Draw bounding box với label
    
    Args:
        box: Dict {'box': [x1,y1,x2,y2], ...} hoặc list [x1,y1,x2,y2,score,...]
        label: Tên class
        score: Confidence score
        color: Màu tùy chỉnh (nếu None thì dùng màu theo class)
    """
    # Parse box format
    if isinstance(box, dict):
        x1, y1, x2, y2 = map(int, box['box'][:4])
    elif isinstance(box, (list, tuple, np.ndarray)):
        x1, y1, x2, y2 = map(int, box[:4])
    else:
        raise ValueError(f"Unknown box format: {type(box)}")

    # Chọn màu theo class
    if color is None:
        color = get_class_color(label)

    # Vẽ box với độ dày dựa trên score
    thickness = 3 if score > 0.7 else 2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Vẽ background cho text
    text = f"{label} {score:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
    
    # Vẽ text (màu trắng)
    cv2.putText(img, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)


def visualize_detections(img, detections, use_class_colors=True, random_colors=False):
    """
    Vẽ bounding boxes lên ảnh
    
    Args:
        img: Image array (BGR)
        detections: List of dicts with keys: 'bbox' (list [x1,y1,x2,y2]), 'label', 'score'
        use_class_colors: Dùng màu theo class từ CLASS_COLORS
        random_colors: Dùng màu ngẫu nhiên cho mỗi object (override use_class_colors)
    
    Returns:
        img: Image với boxes đã vẽ
    """
    result_img = img.copy()
    
    # Generate colors nếu cần
    if random_colors:
        colors = generate_colors(len(detections))
    else:
        colors = [None] * len(detections)  # Sẽ dùng class colors
    
    for i, det in enumerate(detections):
        bbox = det['bbox']
        label = det['label']
        score = det['score']
        color = colors[i] if random_colors else None
        
        draw_box(result_img, bbox, label, score, color)
    
    return result_img


def run_and_visualize(image_path, show=True, save_path="output.jpg"):
    # Load models
    detector, embedder, db_vectors, db_labels = _load_models()
    
    img = cv2.imread(image_path)
    orig_img = img.copy()

    #### ====== TIME: YOLO DETECT ====== ####
    t0 = time.time()
    dets = detector.predict(img, conf_thres=0.05)  # Giảm threshold cho vật nhỏ
    t1 = time.time()
    detect_time = (t1 - t0) * 1000  # ms
    
    print(f"📦 YOLO detected {len(dets)} objects")

    #### ====== TIME: EMBED + MATCH ====== ####
    results = []
    embed_time_total = 0

    for box_data in dets:
        crop = crop_food(img, box_data)
        if crop is None or crop.size == 0:
            continue

        t2 = time.time()
        vec = embedder.embed(crop)
        label, score = find_best_match(vec, db_vectors, db_labels)
        t3 = time.time()

        embed_time_total += (t3 - t2) * 1000

        results.append({
            "bbox": box_data,
            "label": label,
            "score": score
        })

    #### ====== DRAW RESULTS ====== ####
    orig_img = visualize_detections(orig_img, results, use_class_colors=True)

    #### ====== PRINT SUMMARY ====== ####
    print("===================================")
    print(f"YOLO Detect Time: {detect_time:.2f} ms")
    print(f"Embedding+Matching Total: {embed_time_total:.2f} ms")
    if len(results) > 0:
        print(f"Average per object: {embed_time_total / len(results):.2f} ms/object")
    print("Total inference:", detect_time + embed_time_total, "ms")
    print("===================================")

    #### ====== SHOW OR SAVE ====== ####
    if save_path:
        cv2.imwrite(save_path, orig_img)
        print(f"✓ Saved result to: {save_path}")

    if show:
        print("\nPress any key to close, or ESC/Q to quit...")
        cv2.imshow("Result - Press any key or ESC/Q to close", orig_img)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        if key == 27 or key == ord('q') or key == ord('Q'):
            print("Exiting...")

    return results


if __name__ == "__main__":
    import sys
    
    try:
        # Lấy đường dẫn ảnh từ command line hoặc dùng mặc định
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            image_path = "data/images/image_01.jpg"
        
        # Lấy output path (optional) - None nếu không muốn lưu
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f"Processing: {image_path}")
        if output_path:
            print(f"Output will be saved to: {output_path}\n")
        else:
            print("Output will NOT be saved (display only)\n")
        
        run_and_visualize(image_path, show=True, save_path=output_path)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        cv2.destroyAllWindows()
        sys.exit(0)
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        sys.exit(1)
import cv2
import os
from detect import YOLO11Detector
from crop_food import crop_food

REF_DIR = "data/ref_images"
IMG_DIR = "data/images"

# Map số sang tên class (đã xóa candy)
CLASS_MAP = {
    '1': 'coconut',
    '2': 'cua',
    '3': 'macaron',
    '4': 'meden',
    '5': 'melon',
    '0': 'skip'
}

def ensure_class_dirs():
    classes = ["coconut", "cua", "macaron", "meden", "melon"]
    for cls in classes:
        os.makedirs(os.path.join(REF_DIR, cls), exist_ok=True)

def visualize_all_detections(img, detections):
    """Hiển thị tất cả detections trên ảnh để debug"""
    img_vis = img.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['box']
        score = det['score']
        
        # Draw box
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"#{i+1}: {score:.2f}"
        cv2.putText(img_vis, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return img_vis

def generate():
    ensure_class_dirs()

    # Khởi tạo detector với TRAINED MODEL
    print("Loading TRAINED detector...")
    detector = YOLO11Detector("models/yolo11_food_trained.onnx")
    print("Detector loaded!\n")

    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if not img_files:
        print(f"⚠️  No images found in {IMG_DIR}")
        return

    print("=== LEGEND ===")
    print("1: coconut | 2: cua | 3: macaron")
    print("4: meden | 5: melon")
    print("0: Skip current object | ESC: Skip current object")
    print("Q: Quit program | R: Re-detect with lower threshold")
    print("=" * 50 + "\n")

    for img_idx, img_name in enumerate(img_files):
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️  Could not read {img_name}, skipping...")
            continue

        print(f"\n{'='*50}")
        print(f"[{img_idx+1}/{len(img_files)}] Processing: {img_name}")
        print(f"Image size: {img.shape[1]}x{img.shape[0]}px")
        print(f"{'='*50}")

        # Detect với confidence RẤT THẤP vì trained model vẫn cho score thấp với object nhỏ
        conf_threshold = 0.05  # Macaron nhỏ có score ~0.05-0.11
        detections = detector.predict(img, conf_thres=conf_threshold, iou_thres=0.35, min_box_size=5)
        
        # Filter: loại bỏ box không hợp lệ nhưng giữ lại object nhỏ
        filtered_dets = []
        img_h, img_w = img.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            w = x2 - x1
            h = y2 - y1
            score = det['score']
            
            # Skip box không hợp lệ
            if w <= 0 or h <= 0:
                continue
            
            aspect_ratio = max(w/h, h/w) if min(w, h) > 0 else 999
            
            # Filter rules:
            # 1. Loại box quá méo (aspect > 10) - edge artifacts
            # 2. Loại box quá lớn (> 85% ảnh) - background
            # 3. Loại box quá nhỏ (< 80x80) VÀ score thấp (< 0.08) - noise
            # 4. GIỮ LẠI box nhỏ nếu score >= 0.05 (có thể là macaron)
            
            is_too_distorted = aspect_ratio > 10
            is_too_large = w > img_w * 0.85 or h > img_h * 0.85
            is_small_noise = (w < 80 or h < 80) and score < 0.045
            
            if not (is_too_distorted or is_too_large or is_small_noise):
                filtered_dets.append(det)
        
        detections = filtered_dets
        print(f"✓ Found {len(detections)} valid objects (conf >= {conf_threshold}, filtered)")
        
        if len(detections) == 0:
            print("⚠️  No objects detected! Try:")
            print("   - Press 'R' to re-detect with lower threshold")
            print("   - Or press any key to skip this image")
            
            # Show original image
            cv2.imshow("No detections", img)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            
            if key == ord('r') or key == ord('R'):
                # Retry with extremely low threshold
                conf_threshold = 0.02  # Cực thấp để bắt mọi thứ
                print(f"Re-detecting with conf >= {conf_threshold}...")
                detections = detector.predict(img, conf_thres=conf_threshold, iou_thres=0.3, min_box_size=3)
                
                # Apply same filtering
                filtered_dets = []
                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    w = x2 - x1
                    h = y2 - y1
                    aspect_ratio = max(w/h, h/w) if min(w, h) > 0 else 999
                    
                    if aspect_ratio < 10 and w < img_w * 0.85 and h < img_h * 0.85:
                        filtered_dets.append(det)
                
                detections = filtered_dets
                print(f"✓ Found {len(detections)} objects (filtered)")
            else:
                continue

        # Hiển thị overview tất cả detections
        img_overview = visualize_all_detections(img, detections)
        cv2.imshow("All Detections - Press any key to start labeling", img_overview)
        cv2.waitKey(1000)  # Show for 1 second
        cv2.destroyAllWindows()

        # Label từng object
        for i, det in enumerate(detections):
            # Print detection info
            x1, y1, x2, y2 = det['box']
            w = x2 - x1
            h = y2 - y1
            score = det['score']
            print(f"\n  Object {i+1}/{len(detections)}: score={score:.3f}, size={w}x{h}px")

            # Crop object
            crop = crop_food(img, det)
            
            if crop is None or crop.size == 0:
                print("  ⚠️  Failed to crop, skipping...")
                continue

            # Resize crop nếu quá nhỏ để dễ nhìn
            display_crop = crop.copy()
            if display_crop.shape[0] < 100 or display_crop.shape[1] < 100:
                scale = max(100 / display_crop.shape[0], 100 / display_crop.shape[1])
                new_w = int(display_crop.shape[1] * scale)
                new_h = int(display_crop.shape[0] * scale)
                display_crop = cv2.resize(display_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Show crop với context
            window_name = f"Object {i+1}/{len(detections)} - Choose: 1-5 or 0/ESC to skip"
            
            # Add text overlay
            info_text = [
                f"Image: {img_name}",
                f"Object: {i+1}/{len(detections)}",
                f"Score: {score:.3f}",
                f"Size: {w}x{h}px",
                "",
                "1:coconut 2:cua 3:macaron",
                "4:meden 5:melon",
                "0/ESC:skip Q:quit"
            ]
            
            # Create display with info
            text_h = len(info_text) * 25 + 20
            display_with_info = cv2.copyMakeBorder(
                display_crop, text_h, 0, 0, 0,
                cv2.BORDER_CONSTANT, value=(50, 50, 50)
            )
            
            for idx, text in enumerate(info_text):
                cv2.putText(display_with_info, text, (10, 20 + idx * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(window_name, display_with_info)
            
            # Wait for key
            key = cv2.waitKey(0) & 0xFF
            
            # Handle quit
            if key == ord('q') or key == ord('Q'):
                print("\n⚠️  Quitting...")
                cv2.destroyAllWindows()
                return
            
            # Handle skip
            if key == 27 or key == ord('0'):  # ESC or 0
                print("  ⊘ Skipped")
                try:
                    cv2.destroyWindow(window_name)
                except:
                    pass
                continue
            
            # Map key to class
            if 49 <= key <= 53:  # '1' to '5' (đã bỏ candy)
                key_char = chr(key)
                label = CLASS_MAP[key_char]
                
                # Generate unique filename
                existing = len([f for f in os.listdir(os.path.join(REF_DIR, label)) 
                               if f.startswith(img_name.split('.')[0])])
                save_name = f"{img_name.split('.')[0]}_obj{existing}.jpg"
                save_path = os.path.join(REF_DIR, label, save_name)
                
                # Save original crop (not resized)
                cv2.imwrite(save_path, crop)
                print(f"  ✓ Saved as '{label}': {save_name}")
            else:
                print(f"  ⚠️  Invalid key (code: {key}). Skipped.")
            
            try:
                cv2.destroyWindow(window_name)
            except:
                pass  # Window already closed

        print(f"\n✓ Finished processing {img_name}")

    cv2.destroyAllWindows()
    print("\n" + "="*50)
    print("✓ All images processed!")
    print("="*50)

if __name__ == "__main__":
    try:
        generate()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
import os
import shutil

IMAGE_DIR = "data/images"
LABEL_DIR = "data/labels"
OUTPUT_DIR = "data/yolov11_dataset"

def prepare_dataset():
    """
    Chia dataset 90% train / 10% val
    Chọn validation images theo thứ tự để đảm bảo đại diện cho tất cả setup camera
    """
    images = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg") or f.endswith(".png")])
    total = len(images)
    
    # 90% train, 10% val
    val_count = max(1, int(total * 0.1))  # Ít nhất 1 ảnh val
    train_count = total - val_count
    
    print(f"Total images: {total}")
    print(f"Train: {train_count} images (90%)")
    print(f"Val: {val_count} images (10%)")
    
    # Chọn val images đều đặn trong dataset để đại diện cho các setup khác nhau
    # VD: với 25 ảnh, chọn mỗi 10 ảnh (25/3 ≈ 8) → ảnh 8, 16, 24
    step = total // val_count if val_count > 0 else total
    val_indices = [i * step for i in range(1, val_count + 1)]
    if val_indices[-1] >= total:
        val_indices[-1] = total - 1
    
    val_imgs = [images[i] for i in val_indices]
    train_imgs = [img for img in images if img not in val_imgs]
    
    print(f"\nValidation images: {val_imgs}")

    folders = [
        "images/train", "images/val",
        "labels/train", "labels/val"
    ]

    for folder in folders:
        os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)

    def copy_files(file_list, split):
        for img_name in file_list:
            lbl_name = img_name.replace(".jpg", ".txt").replace(".png", ".txt")
            
            img_src = os.path.join(IMAGE_DIR, img_name)
            lbl_src = os.path.join(LABEL_DIR, lbl_name)
            
            # Kiểm tra file tồn tại
            if not os.path.exists(img_src):
                print(f"⚠️  Image not found: {img_name}")
                continue
            if not os.path.exists(lbl_src):
                print(f"⚠️  Label not found: {lbl_name}")
                continue

            shutil.copy(img_src, os.path.join(OUTPUT_DIR, f"images/{split}", img_name))
            shutil.copy(lbl_src, os.path.join(OUTPUT_DIR, f"labels/{split}", lbl_name))

    copy_files(train_imgs, "train")
    copy_files(val_imgs, "val")

    print(f"\n✓ Dataset prepared successfully!")
    print(f"✓ Train: {len(train_imgs)} images → {OUTPUT_DIR}/images/train/")
    print(f"✓ Val: {len(val_imgs)} images → {OUTPUT_DIR}/images/val/")

if __name__ == "__main__":
    prepare_dataset()

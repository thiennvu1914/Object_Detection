import os
import cv2
import numpy as np
from embed import MobileCLIP2Embedder

REF_DIR = "data/ref_images"
DB_DIR = "db"
VEC_OUT = os.path.join(DB_DIR, "vectors.npy")
LAB_OUT = os.path.join(DB_DIR, "labels.txt")

def build_db():
    os.makedirs(DB_DIR, exist_ok=True)
    embedder = MobileCLIP2Embedder()

    vectors = []
    labels = []

    for label in os.listdir(REF_DIR):
        label_dir = os.path.join(REF_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            vec = embedder.embed(img)
            vectors.append(vec)
            labels.append(label)

    np.save(VEC_OUT, np.array(vectors))
    open(LAB_OUT, "w").write("\n".join(labels))

    print(f"✓ Database built successfully with {len(vectors)} vectors.")

if __name__ == "__main__":
    build_db()
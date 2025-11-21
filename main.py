"""
MAIN PIPELINE: Food Detection + Classification
==============================================
Detect objects → Crop → Embedding → Classification → Visualize

Usage:
    python main.py <image_path>
    
Example:
    python main.py data/images/image_01.jpg
"""
import sys
import time
import cv2
import numpy as np
from pathlib import Path
import torch

# Add directories to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import from food_detection package
from food_detection.core.detector import YOLOEFoodDetector
from food_detection.core.embedder import MobileCLIPEmbedder  
from food_detection.utils.visualize import visualize_detections, get_class_color


class FoodDetectionPipeline:
    """Full pipeline from detection to classification"""
    
    def __init__(self):
        print("="*70)
        print("INITIALIZING FOOD DETECTION PIPELINE")
        print("="*70)
        
        # Load models
        print("\n[1/3] Loading YOLOE detector...")
        t0 = time.time()
        self.detector = YOLOEFoodDetector()
        print(f"      ✓ YOLOE loaded in {time.time()-t0:.2f}s")
        
        print("\n[2/3] Loading MobileCLIP embedder...")
        t0 = time.time()
        self.embedder = MobileCLIPEmbedder()
        print(f"      ✓ MobileCLIP loaded in {time.time()-t0:.2f}s")
        
        print("\n[3/3] Loading reference embeddings...")
        t0 = time.time()
        self.ref_embeddings = self.load_reference_embeddings()
        if len(self.ref_embeddings) == 0:
            print(f"      ⚠️  No reference images found in data/ref_images/")
        else:
            print(f"      ✓ Loaded {len(self.ref_embeddings)} reference embeddings in {time.time()-t0:.2f}s")
            classes = set(item['class'] for item in self.ref_embeddings)
            print(f"      → Classes: {', '.join(sorted(classes))}")
            
            # Import để có thể kiểm tra màu
            from food_detection.utils.visualize import get_class_color
            for cls in sorted(classes):
                color = get_class_color(cls)
                print(f"         • {cls}: RGB{color[::-1]}")
        
        print("\n" + "="*70)
        print("READY TO PROCESS IMAGES")
        print("="*70 + "\n")
    
    def load_reference_embeddings(self):
        """Load reference embeddings from data/ref_images/"""
        ref_dir = Path("data/ref_images")
        if not ref_dir.exists():
            return []
        
        ref_embeddings = []
        for class_dir in ref_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue
            
            class_name = class_dir.name
            for img_file in class_dir.glob("*.jpg"):
                try:
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    embedding = self.embedder.embed(img)
                    ref_embeddings.append({
                        'class': class_name,
                        'image': img_file.name,
                        'embedding': embedding
                    })
                except Exception as e:
                    print(f"      ⚠️  Failed to load {img_file}: {e}")
        
        return ref_embeddings
    
    def classify(self, query_embedding):
        """Classify using cosine similarity with reference embeddings"""
        if len(self.ref_embeddings) == 0:
            return 'unknown', 0.0
        
        # Normalize query embedding
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Compute similarities with all references
        best_class = 'unknown'
        best_similarity = 0.0
        
        class_similarities = {}  # Average similarity per class
        
        for ref in self.ref_embeddings:
            ref_norm = ref['embedding'] / (np.linalg.norm(ref['embedding']) + 1e-8)
            similarity = float(np.dot(query_norm, ref_norm))
            
            class_name = ref['class']
            if class_name not in class_similarities:
                class_similarities[class_name] = []
            class_similarities[class_name].append(similarity)
        
        # Get class with highest average similarity
        for class_name, similarities in class_similarities.items():
            avg_sim = np.mean(similarities)
            if avg_sim > best_similarity:
                best_similarity = avg_sim
                best_class = class_name
        
        return best_class, best_similarity
    
    def process_image(self, image_path):
        """
        Full pipeline:
        1. Detect objects
        2. Crop objects
        3. Generate embeddings
        4. Classify (if DB available)
        5. Visualize results
        """
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return None
        
        print(f"Processing: {image_path.name}")
        print("-" * 70)
        
        # Read image
        img = cv2.imread(str(image_path))
        img_h, img_w = img.shape[:2]
        
        # ============================================================
        # STAGE 1: DETECTION
        # ============================================================
        print("\n[STAGE 1] Object Detection")
        t0 = time.time()
        
        result = self.detector.predict(str(image_path))
        food_boxes = self.detector.ensemble_filter(result)
        
        detection_time = time.time() - t0
        print(f"           ✓ Detected {len(food_boxes)} objects in {detection_time:.3f}s")
        print(f"           → Performance: {1000*detection_time:.0f}ms")
        
        if len(food_boxes) == 0:
            print("\n⚠️  No food items detected!")
            return {
                'image_path': image_path,
                'detections': [],
                'total_time': detection_time
            }
        
        # ============================================================
        # STAGE 2: CROP OBJECTS
        # ============================================================
        print("\n[STAGE 2] Crop Objects")
        t0 = time.time()
        
        crops = []
        for i, box in enumerate(food_boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            
            # Crop with padding
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img_w, x2 + pad)
            y2 = min(img_h, y2 + pad)
            
            crop = img[y1:y2, x1:x2]
            crops.append({
                'id': i,
                'crop': crop,
                'bbox': [x1, y1, x2, y2],
                'conf': box.conf[0].item(),
                'original_class': self.detector.model.names[int(box.cls[0].item())]
            })
        
        crop_time = time.time() - t0
        print(f"           ✓ Cropped {len(crops)} objects in {crop_time:.3f}s")
        print(f"           → Performance: {1000*crop_time/len(crops):.0f}ms per crop")
        
        # ============================================================
        # STAGE 3: GENERATE EMBEDDINGS
        # ============================================================
        print("\n[STAGE 3] Generate Embeddings")
        t0 = time.time()
        
        embeddings = []
        for crop_data in crops:
            emb = self.embedder.embed(crop_data['crop'])
            embeddings.append(emb)
            crop_data['embedding'] = emb
        
        embedding_time = time.time() - t0
        print(f"           ✓ Generated {len(embeddings)} embeddings in {embedding_time:.3f}s")
        print(f"           → Performance: {1000*embedding_time/len(embeddings):.0f}ms per embedding")
        print(f"           → Embedding dim: {embeddings[0].shape}")
        
        # ============================================================
        # STAGE 4: CLASSIFICATION
        # ============================================================
        print("\n[STAGE 4] Classification")
        t0 = time.time()
        
        if len(self.ref_embeddings) == 0:
            print("           ⚠️  Skipped (no reference embeddings)")
            classification_time = 0
            for crop_data in crops:
                crop_data['predicted_class'] = 'unknown'
                crop_data['similarity'] = 0.0
        else:
            for crop_data in crops:
                pred_class, similarity = self.classify(crop_data['embedding'])
                crop_data['predicted_class'] = pred_class
                crop_data['similarity'] = similarity
            
            classification_time = time.time() - t0
            print(f"           ✓ Classified {len(crops)} objects in {classification_time:.3f}s")
            print(f"           → Performance: {1000*classification_time/len(crops):.0f}ms per classification")
        
        # ============================================================
        # STAGE 5: VISUALIZE
        # ============================================================
        print("\n[STAGE 5] Visualize Results")
        t0 = time.time()
        
        # Prepare detections for visualize_detections function
        detections = []
        for crop_data in crops:
            detections.append({
                'bbox': crop_data['bbox'],
                'label': crop_data['predicted_class'],
                'score': crop_data['similarity'] if len(self.ref_embeddings) > 0 else crop_data['conf']
            })
        
        # Use visualize_detections from visualize.py with class colors
        result_img = visualize_detections(img, detections, use_class_colors=True, random_colors=False)
        
        # Save result
        output_dir = Path("outputs/pipeline")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{image_path.stem}_result.jpg"
        cv2.imwrite(str(output_path), result_img)
        
        visualize_time = time.time() - t0
        print(f"           ✓ Saved to: {output_path}")
        print(f"           → Visualization time: {visualize_time:.3f}s")
        
        # ============================================================
        # SUMMARY
        # ============================================================
        total_time = detection_time + crop_time + embedding_time + classification_time + visualize_time
        
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70)
        print(f"Total objects detected: {len(crops)}")
        print(f"\nTiming breakdown (milliseconds):")
        print(f"  1. Detection:       {detection_time*1000:>8.1f}ms  ({100*detection_time/total_time:>5.1f}%)")
        print(f"  2. Cropping:        {crop_time*1000:>8.1f}ms  ({100*crop_time/total_time:>5.1f}%)")
        print(f"  3. Embedding:       {embedding_time*1000:>8.1f}ms  ({100*embedding_time/total_time:>5.1f}%)")
        print(f"  4. Classification:  {classification_time*1000:>8.1f}ms  ({100*classification_time/total_time:>5.1f}%)")
        print(f"  5. Visualization:   {visualize_time*1000:>8.1f}ms  ({100*visualize_time/total_time:>5.1f}%)")
        print(f"  " + "-"*40)
        print(f"  TOTAL:              {total_time*1000:>8.1f}ms ({total_time:.3f}s)")
        print(f"\nThroughput: {len(crops)/total_time:.2f} objects/second")
        print(f"Average per object: {1000*total_time/len(crops):.1f}ms")
        print("="*70)
        
        # Print detection results
        print("\n📋 DETECTION RESULTS:")
        for i, crop_data in enumerate(crops, 1):
            print(f"  {i}. {crop_data['predicted_class']:>15s} "
                  f"(similarity: {crop_data['similarity']:.3f}, "
                  f"conf: {crop_data['conf']:.3f}, "
                  f"bbox: {crop_data['bbox']})")
        
        return {
            'image_path': image_path,
            'output_path': output_path,
            'detections': crops,
            'timing': {
                'detection': detection_time,
                'crop': crop_time,
                'embedding': embedding_time,
                'classification': classification_time,
                'visualization': visualize_time,
                'total': total_time
            },
            'performance': {
                'fps': len(crops) / total_time,
                'ms_per_object': 1000 * total_time / len(crops) if len(crops) > 0 else 0
            }
        }


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("""
Food Detection Pipeline
========================

Usage:
    python main.py <image_path>

Example:
    python main.py data/images/image_01.jpg
    python main.py data/images/image_25.jpg

Pipeline stages:
    1. Object Detection (YOLOE)
    2. Crop Objects
    3. Generate Embeddings (MobileCLIP)
    4. Classification (cosine similarity)
    5. Visualize Results

Output:
    - Result image: outputs/pipeline/<image_name>_result.jpg
    - Performance metrics logged to console
        """)
        return
    
    image_path = sys.argv[1]
    
    # Initialize pipeline
    pipeline = FoodDetectionPipeline()
    
    # Process image
    result = pipeline.process_image(image_path)
    
    if result:
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Processed {len(result['detections'])} objects in {result['timing']['total']*1000:.1f}ms ({result['timing']['total']:.3f}s)")
        print(f"🖼️  Result saved to: {result['output_path']}")
        
        # Display result in window
        print(f"\n👁️  Displaying result... (Press any key to close)")
        result_img = cv2.imread(str(result['output_path']))
        if result_img is not None:
            # Resize if image too large
            h, w = result_img.shape[:2]
            max_display_size = 1200
            if max(h, w) > max_display_size:
                scale = max_display_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                result_img = cv2.resize(result_img, (new_w, new_h))
            
            cv2.imshow('Food Detection Result', result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

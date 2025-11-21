# Food Object Detection & Classification

## Overview
Complete pipeline for detecting and classifying food items in images.

## Architecture
1. **Detection**: YOLO11n (ONNX) - Single-class food detector
2. **Classification**: MobileCLIP S2 - 5-class fine-grained classifier
3. **Matching**: Cosine similarity on 512-dim embeddings

## Key Implementation Details

### Coordinate Format
YOLO11 ONNX outputs center-format coordinates:
```python
[x_center, y_center, width, height, class_scores...]
```
Convert to corners:
```python
x1 = x_center - width / 2
y1 = y_center - height / 2
x2 = x_center + width / 2
y2 = y_center + height / 2
```

### Small Object Detection
- Confidence threshold: 0.05 (macaron scores ~0.054-0.114)
- Mosaic augmentation: 1.0
- MixUp augmentation: 0.1
- Min box size: 10px
- Aspect ratio filtering in generate_ref_images.py

### Cropping Strategy
- Padding: 5px around detected box
- Auto-resize if crop < 80px
- Boundary validation to prevent out-of-bounds

### Color Scheme (BGR)
```python
CLASS_COLORS = {
    'coconut': (0, 255, 0),    # Green
    'cua': (0, 0, 255),        # Red
    'macaron': (255, 0, 0),    # Blue
    'meden': (0, 255, 255),    # Yellow
    'melon': (255, 0, 255)     # Magenta
}
```

## Training Notes

### Best Model Selection
- Trained for 50 epochs
- Best mAP50 at epoch 9: 0.736
- Early stopping triggered at epoch 24
- AdamW optimizer with box_loss=7.5

### Dataset Split
- Train: 70% (10 images)
- Val: 30% (4 images)
- All labels converted to class 0 (single-class detector)

## Performance Metrics

### Inference Time
- YOLO Detection: ~134ms
- Crop + Resize: ~5-10ms per box
- MobileCLIP Embedding: ~813ms per crop
- Total Pipeline: ~947ms per image

### Model Sizes
- YOLO11n ONNX: 5.2 MB
- MobileCLIP S2: 39 MB

## Dependencies

### Critical Version Requirements
- Python: 3.11.8 (onnxruntime incompatible with 3.13+)
- onnxruntime: Latest (requires Python ≤3.12)
- torch: 2.5.1+cu124
- ultralytics: Latest

### Installation Order
1. Create Python 3.11 environment
2. Install torch with CUDA
3. Install onnxruntime
4. Install MobileCLIP from GitHub (bypasses version check)
5. Install remaining dependencies

## Known Issues

### Fixed Issues
1. **sklearn import error**: Package name is `scikit-learn`, import as `sklearn`
2. **YAML path duplication**: Fixed relative paths in train_yolo.py
3. **Center-format coordinates**: YOLO11 outputs center format, not corners
4. **Small object detection**: Lowered conf from 0.15 to 0.05
5. **Keyboard interrupt**: Added exception handling in visualize.py

### Limitations
- Pipeline too slow for real-time (target: 150-200ms, current: ~947ms)
- MobileCLIP embedding is bottleneck (~813ms)
- Small dataset (14 images total)

## Future Optimizations

1. **Speed Improvements**:
   - Batch embedding inference
   - TensorRT conversion
   - Multi-threading for crops
   - Model quantization

2. **Accuracy Improvements**:
   - Larger training dataset
   - Data augmentation expansion
   - Test-time augmentation
   - Ensemble methods

3. **Features**:
   - Real-time video processing
   - Multi-object tracking
   - Confidence-based filtering
   - Database search optimization
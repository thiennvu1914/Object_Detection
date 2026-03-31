# 🍱 Food Detection System

Real-time food detection với YOLOE, MobileCLIP, và SSIM-based Change Detection.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## ✨ Features

- **🎯 Smart Detection**: YOLOE + Ensemble filtering (spatial, size, ML)
- **🔍 High Accuracy**: MobileCLIP 512-dim embeddings, cosine similarity
- **⚡ Change Detection**: SSIM layer giảm 90-95% YOLOE calls (ROI-based, high sensitivity)
- **🚀 Real-time**: ~1s/image, streaming với anti-lag optimization
- **🗄️ Database**: SQLite cache embeddings & detection history
- **🌐 REST API**: FastAPI + WebSocket dual-stream
- **📹 Demo**: test_streaming_demo.py với ROI visualization

## 🎬 Demo Video

[Watch the demo](https://drive.google.com/drive/folders/1cSzTw8m8LVakwSYDJg0ILTHTlpNp12N2?usp=drive_link)

## 📦 Quick Start

```powershell
# 1. Setup environment
python -m venv .venv11
.\.venv11\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Download models (~451 MB total) - REQUIRED
# See models/DOWNLOAD_MODELS.md for detailed guide
python -c "from ultralytics import YOLO; import shutil; model = YOLO('yolo11l-seg.pt'); shutil.copy(model.ckpt_path, 'models/yoloe-11l-seg-pf.pt')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('apple/MobileCLIP-S2-OpenCLIP', local_dir='models/mobileclip_s2')"

# 3. Verify & Run
python -c "from food_detection import FoodDetectionPipeline; pipeline = FoodDetectionPipeline(); print('✓ Setup complete')"
python main.py data/images/image_01.jpg              # CLI
python run_api.py                                     # API (localhost:8000)
python tests/test_streaming_demo.py                  # Real-time demo with ROI
```

**📖 Detailed model installation:** See [models/DOWNLOAD_MODELS.md](models/DOWNLOAD_MODELS.md)

## 📁 Project Structure

```
ObjectDetection/
├── food_detection/           # Main package
│   ├── core/                # YOLOE, MobileCLIP, Pipeline
│   ├── streaming/           # Change Detector (SSIM), Camera, WebSocket
│   ├── api/                 # FastAPI routes
│   └── database.py          # SQLite manager
├── tests/                   # Test suite
│   ├── test_streaming_demo.py  # Real-time demo with ROI
│   ├── test_change_detection.py
│   └── test_*.py
├── static/                  # HTML demos
├── data/
│   ├── images/             # Test images
│   └── ref_images/         # 5 classes (coconut, cua, macaron, meden, melon)
├── models/                 # YOLOE (70MB) + MobileCLIP (380MB)
└── main.py, run_api.py     # Entry points
```

## 🎨 Supported Classes

**5 classes**: `coconut`, `cua`, `macaron`, `meden`, `melon` (6 ref images mỗi class)

**Thêm class mới:** Tạo folder `data/ref_images/<class_name>/`, thêm 5-10 ảnh reference

## 🔧 Usage

### CLI
```powershell
python main.py image.jpg --conf 0.5
```

### API
```python
import requests
response = requests.post("http://localhost:8000/api/v1/detect", 
                        files={"file": open("image.jpg", "rb")})
print(response.json())
```

### Python Package
```python
from food_detection import FoodDetectionPipeline
pipeline = FoodDetectionPipeline()
result = pipeline.process_image("image.jpg")
```

## 🧠 Architecture

**Pipeline**: YOLOE Detection → Ensemble Filtering → MobileCLIP Embedding → Classification
**Streaming**: ROI-based Change Detection (SSIM 0.94, Diff 0.05) → 90-95% YOLOE reduction

**Performance**: ~2s/image (Detection 1.3s, Embedding 1s)

## 📊 API Response

```json
{
  "success": true,
  "data": {
    "detections": [{"bbox": [x1,y1,x2,y2], "class": "melon", "similarity": 0.889}],
    "count": 5,
    "processing_time": 2.411
  }
}
```

## 🚀 Change Detection (Streaming)

**ROI-based SSIM + Frame Difference** → giảm 90-95% YOLOE calls

```python
from food_detection.streaming.change_detector import ChangeDetector

# High sensitivity for tray area (center 70%)
detector = ChangeDetector(
    ssim_threshold=0.94,    # Detect 6% structural change
    diff_threshold=0.05,    # Detect 5% pixel change
    resize_height=360,      # Top-down camera optimized
    roi=(96, 72, 544, 408)  # Focus on tray, ignore background
)

# Test streaming demo
python tests/test_streaming_demo.py  # Real-time with ROI visualization
```

**📖 Full config:** See `STREAMING_CONFIG.md`

---

## 🙏 Credits

- **YOLOE**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **MobileCLIP**: [Apple ML Research](https://github.com/apple/ml-mobileclip)

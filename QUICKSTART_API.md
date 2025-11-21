# Quick Start Guide - API Mode

## 📦 Cài Đặt

### 1. Clone và setup

```powershell
git clone https://github.com/thiennvu1914/ObjectDetection.git
cd ObjectDetection
git checkout feature/modularize-source-code

# Tạo virtual environment
python -m venv .venv11
.\.venv11\Scripts\Activate.ps1

# Cài dependencies
pip install -r requirements.txt
```

### 2. Tải models

```powershell
# Tải MobileCLIP
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('apple/MobileCLIP-S2-OpenCLIP', local_dir='models/mobileclip_s2')"

# Tải YOLOE (thủ công)
# Download từ: https://github.com/ultralytics/assets/releases/
# Đặt vào: models/yoloe-11l-seg-pf.pt
```

### 3. Chạy API Server

```powershell
python run_api.py
```

Server khởi động tại: **http://localhost:8000**

---

## 🔌 Sử Dụng API

### Option 1: Swagger UI (Web Interface)

Mở trình duyệt: **http://localhost:8000/docs**

### Option 2: Python Client

```python
from example_client import FoodDetectionClient

client = FoodDetectionClient()

# Detect single image
result = client.detect("data/images/image_01.jpg", confidence=0.5)
print(f"Found {result['data']['count']} items")

# Get available classes
classes = client.get_classes()
print(classes['data']['classes'])
```

### Option 3: cURL

```powershell
curl -X POST "http://localhost:8000/api/v1/detect?confidence=0.5" -F "file=@data/images/image_01.jpg"
```

### Option 4: From Another App

```python
import requests

# From your Django/Flask/etc app
def detect_food(image_file_path):
    with open(image_file_path, 'rb') as f:
        response = requests.post(
            "http://localhost:8000/api/v1/detect",
            files={"file": f},
            params={"confidence": 0.5}
        )
    return response.json()

# Usage
result = detect_food("path/to/food.jpg")
for item in result['data']['detections']:
    print(f"{item['class']}: {item['similarity']:.2f}")
```

---

## 📡 API Endpoints

### `POST /api/v1/detect`
Phát hiện món ăn trong 1 ảnh

### `POST /api/v1/detect-batch`
Xử lý nhiều ảnh cùng lúc (tối đa 10)

### `GET /api/v1/classes`
Lấy danh sách classes có sẵn

---

## 🔧 Module Hóa

Project đã được tổ chức theo cấu trúc module:

```
food_detection/
├── __init__.py          # Package entry
├── core/                # Core functionality
│   ├── detector.py     # YOLOE detection
│   ├── embedder.py     # MobileCLIP embeddings
│   ├── classifier.py   # Classification logic
│   └── pipeline.py     # End-to-end pipeline
├── api/                 # FastAPI application
│   ├── app.py          # FastAPI app
│   └── routes.py       # API endpoints
└── utils/               # Utilities
    ├── image.py        # Image processing
    └── visualize.py    # Drawing functions
```

### Import trong Python

```python
# Import pipeline
from food_detection import FoodDetectionPipeline

# Import individual modules
from food_detection.core import YOLOEFoodDetector, MobileCLIPEmbedder
from food_detection.utils import visualize_detections

# Use in your code
pipeline = FoodDetectionPipeline()
result = pipeline.process_image("image.jpg")
```

---

## 🚀 Production Deployment

```powershell
# Install production server
pip install gunicorn

# Run with multiple workers
gunicorn food_detection.api.app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 📝 Docs

- **API Documentation**: `API_README.md`
- **Project README**: `README.md`
- **Example Client**: `example_client.py`

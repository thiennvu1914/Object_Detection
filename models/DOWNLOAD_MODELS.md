# 📦 Model Installation Guide

Models không được include trong Git repo do kích thước lớn. Follow guide này để download và setup.

---

## 📋 Prerequisites

```powershell
# 1. Tạo virtual environment (Python 3.11+)
python -m venv .venv11
.\.venv11\Scripts\Activate.ps1

# 2. Cài dependencies cơ bản
pip install torch torchvision ultralytics
pip install git+https://github.com/apple/ml-mobileclip.git
pip install huggingface_hub  # For MobileCLIP download
```

---

## 🎯 Required Models

### 1️⃣ YOLOE Model (~71 MB)

**File:** `yoloe-11l-seg-pf.pt`  
**Purpose:** Object detection and segmentation (prompt-free)

**Option A: Auto-download (RECOMMENDED)**
```powershell
# Tự động tải và đặt đúng vị trí
python -c "from ultralytics import YOLO; model = YOLO('yolo11l-seg.pt'); import shutil; shutil.copy(model.ckpt_path, 'models/yoloe-11l-seg-pf.pt')"

# Verify
Test-Path models\yoloe-11l-seg-pf.pt  # Should return: True
```

**Option B: Manual download**
```powershell
# 1. Download từ Ultralytics
# URL: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11l-seg.pt

# 2. Đổi tên và đặt vào models/
Rename-Item -Path "Downloads\yolo11l-seg.pt" -NewName "yoloe-11l-seg-pf.pt"
Move-Item "Downloads\yoloe-11l-seg-pf.pt" models\

# 3. Verify
Get-Item models\yoloe-11l-seg-pf.pt | Select-Object Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}}
# Expected: ~70.82 MB
```

---

### 2️⃣ MobileCLIP Model (~380 MB)

**File:** `mobileclip_s2/` (folder chứa nhiều files)  
**Purpose:** Generate 512-dim embeddings cho food classification

**Option A: HuggingFace Hub (RECOMMENDED)**
```powershell
# Tự động tải toàn bộ model folder
python -c "from huggingface_hub import snapshot_download; snapshot_download('apple/MobileCLIP-S2-OpenCLIP', local_dir='models/mobileclip_s2', ignore_patterns=['*.bin', '*.safetensors'])"

# Verify
Test-Path models\mobileclip_s2\open_clip_pytorch_model.bin  # True
Get-ChildItem models\mobileclip_s2
```

**Option B: Git LFS (nếu có git-lfs)**
```powershell
# Install git-lfs nếu chưa có
# Download: https://git-lfs.github.com/

git lfs install
git clone https://huggingface.co/apple/MobileCLIP-S2-OpenCLIP models/mobileclip_s2
```

**Option C: Manual từ HuggingFace web**
1. Truy cập: https://huggingface.co/apple/MobileCLIP-S2-OpenCLIP/tree/main
2. Download files:
   - `open_clip_pytorch_model.bin` (380 MB) - **BẮT BUỘC**
   - `open_clip_config.json`
   - `preprocessor_config.json`
   - `tokenizer_config.json` (optional)
3. Tạo folder và copy files:
```powershell
New-Item -ItemType Directory -Force models\mobileclip_s2
Move-Item Downloads\open_clip_*.* models\mobileclip_s2\
```

---

## ✅ Verification & Testing

### Check file structure
```powershell
# List all model files
Get-ChildItem models -Recurse -File | Select-Object FullName, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}}

# Expected structure:
# models/
# ├── yoloe-11l-seg-pf.pt                          (~71 MB)
# └── mobileclip_s2/
#     ├── open_clip_pytorch_model.bin              (~380 MB)
#     ├── open_clip_config.json
#     └── preprocessor_config.json
```

### Test models load correctly
```powershell
# Test YOLOE
python -c "from ultralytics import YOLO; model = YOLO('models/yoloe-11l-seg-pf.pt'); print('✓ YOLOE loaded')"

# Test MobileCLIP
python -c "from food_detection.core.embedder import MobileCLIPEmbedder; embedder = MobileCLIPEmbedder('models/mobileclip_s2'); print('✓ MobileCLIP loaded')"

# Full pipeline test
python main.py data/images/image_01.jpg
# Should output: Detected N items with classes
```

---

## 🐛 Troubleshooting

### ❌ Error: `FileNotFoundError: models/yoloe-11l-seg-pf.pt`
```powershell
# Check file tồn tại
Test-Path models\yoloe-11l-seg-pf.pt

# Nếu False, tải lại:
python -c "from ultralytics import YOLO; model = YOLO('yolo11l-seg.pt'); import shutil; shutil.copy(model.ckpt_path, 'models/yoloe-11l-seg-pf.pt')"
```

### ❌ Error: `No module named 'mobileclip'`
```powershell
# Cài MobileCLIP từ source
pip install git+https://github.com/apple/ml-mobileclip.git

# Verify
python -c "import mobileclip; print('OK')"
```

### ❌ Error: `MobileCLIP model not found`
```powershell
# Check folder structure
Test-Path models\mobileclip_s2\open_clip_pytorch_model.bin

# Nếu False, tải lại:
python -c "from huggingface_hub import snapshot_download; snapshot_download('apple/MobileCLIP-S2-OpenCLIP', local_dir='models/mobileclip_s2')"
```

### ❌ Error: `torch.cuda.OutOfMemoryError`
```powershell
# Giảm batch size hoặc dùng CPU
# Edit food_detection/core/detector.py:
# self.model = YOLO(model_path, device='cpu')  # Force CPU
```

### ❌ Error: `RuntimeError: Couldn't load custom C++ ops`
```powershell
# Reinstall PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📊 Model Sizes & Performance

| Model | Size | Load Time | Inference Time |
|-------|------|-----------|----------------|
| YOLOE | 71 MB | ~2s | ~1.3s/image |
| MobileCLIP | 380 MB | ~3s | ~1s/image |
| **Total** | **451 MB** | **~5s** | **~2.4s/image** |

**Note**: Load time chỉ xảy ra 1 lần khi khởi động. Inference time là per image.

---

## 🚀 Quick Setup Script

**All-in-one setup:**
```powershell
# Run this to download all models automatically
python -c "
# YOLOE
from ultralytics import YOLO
import shutil
model = YOLO('yolo11l-seg.pt')
shutil.copy(model.ckpt_path, 'models/yoloe-11l-seg-pf.pt')
print('✓ YOLOE downloaded')

# MobileCLIP
from huggingface_hub import snapshot_download
snapshot_download('apple/MobileCLIP-S2-OpenCLIP', local_dir='models/mobileclip_s2')
print('✓ MobileCLIP downloaded')
"

# Verify
python -c "from food_detection import FoodDetectionPipeline; pipeline = FoodDetectionPipeline(); print('✓ All models loaded successfully')"
```

---

## 📝 Notes

- Models được `.gitignore` để tránh bloat repo size
- Mỗi developer cần tải models riêng sau khi clone repo
- Cache embeddings được lưu trong `food_detection.db` (auto-created)
- Models tương thích với: **Python 3.11+, PyTorch 2.0+, CUDA 11.8+**

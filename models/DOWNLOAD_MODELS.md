# Model Downloads

This directory contains large model files that are **NOT** included in the Git repository due to their size.

## Required Models

### 1. YOLOE Model (70.8 MB)
**File:** `yoloe-11l-seg-pf.pt`

**Download:**
```bash
# Download YOLOE-11L-Seg-PF from Ultralytics
# Option 1: Auto-download via code
python -c "from ultralytics import YOLO; YOLO('yolo11l-seg.pt')"

# Option 2: Manual download
# Visit: https://github.com/ultralytics/assets/releases/
# Download: yoloe-11l-seg-pf.pt
# Place in: models/yoloe-11l-seg-pf.pt
```

**Purpose:** Prompt-free object detection and segmentation

---

### 2. MobileCLIP Model (379.6 MB)
**File:** `mobileclip_s2/mobileclip_s2.pt`

**Download:**
```bash
# Download from Hugging Face
mkdir -p models/mobileclip_s2
cd models/mobileclip_s2

# Download using Python
python -c "
from mobileclip import get_mobileclip_model
import torch

model, _, preprocess = get_mobileclip_model('mobileclip_s2')
torch.save(model.state_dict(), 'mobileclip_s2.pt')
"
```

**Alternative:** Clone from Hugging Face Hub
```bash
git lfs install
git clone https://huggingface.co/apple/MobileCLIP-S2-OpenCLIP models/mobileclip_s2
```

**Purpose:** Generate 512-dimensional embeddings for food classification

---

## Optional Models

### YOLO11n Trained Model (5.2 MB)
**File:** `yolo11_food_trained.onnx`

**Purpose:** Legacy YOLO11 model trained on food dataset (optional)

**Note:** The main pipeline uses YOLOE, this is for backward compatibility only.

---

## Directory Structure

After downloading, your `models/` directory should look like:

```
models/
├── .gitkeep
├── DOWNLOAD_MODELS.md          # This file
├── yoloe-11l-seg-pf.pt         # 70.8 MB - YOLOE model
├── yolo11_food_trained.onnx    # 5.2 MB - Legacy YOLO11 (optional)
└── mobileclip_s2/              # MobileCLIP directory
    ├── mobileclip_s2.pt        # 379.6 MB - MobileCLIP weights
    ├── config.json
    ├── LICENSE
    └── README.md
```

---

## Verification

After downloading, verify the models:

```bash
# Check file sizes
Get-ChildItem models -Recurse -File | Select-Object Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}}

# Expected output:
# yoloe-11l-seg-pf.pt      : 70.82 MB
# mobileclip_s2.pt         : 379.63 MB
```

---

## Troubleshooting

### YOLOE Model Not Found
```python
# Error: FileNotFoundError: models/yoloe-11l-seg-pf.pt not found

# Solution: Download the model
from ultralytics import YOLO
model = YOLO('yolo11l-seg.pt')  # Auto-downloads to cache
# Then copy to models/yoloe-11l-seg-pf.pt
```

### MobileCLIP Import Error
```python
# Error: No module named 'mobileclip'

# Solution: Install MobileCLIP
pip install git+https://github.com/apple/ml-mobileclip.git
```

---

## Git LFS (Optional)

If you want to track large models with Git LFS:

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.pt"
git lfs track "*.onnx"

# Add to .gitattributes
echo "*.pt filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
echo "*.onnx filter=lfs diff=lfs merge=lfs -text" >> .gitattributes
```

**Note:** This repo uses `.gitignore` to exclude models by default to avoid large repo size.

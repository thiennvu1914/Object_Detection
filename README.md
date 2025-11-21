# 🍱 Food Object Detection & Classification

Phát hiện và phân loại món ăn trong khay sử dụng YOLOE và MobileCLIP.

**Tính năng:**
- 🎯 Phát hiện thông minh (loại bỏ background, khung trùng lặp)
- 🔍 Phân loại dựa trên embeddings 512-dim  
- 🎨 Màu sắc cố định cho từng class
- ⚡ ~2s/ảnh

---

## 📦 Cài Đặt

### 1. Clone repo

```powershell
git clone https://github.com/thiennvu1914/ObjectDetection.git
cd ObjectDetection
```

### 2. Tạo virtual environment

```powershell
python -m venv .venv11
.\.venv11\Scripts\Activate.ps1
```

### 3. Cài dependencies

```powershell
pip install -r requirements.txt
```

**Yêu cầu:** Python 3.11+, Windows 10/11, RAM 8GB+

---

## 📥 Tải Models

**⚠️ Models không có trong repo, phải tải riêng!**

### Cách 1: Tự động (Khuyên dùng)

```powershell
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('apple/MobileCLIP-S2-OpenCLIP', local_dir='models/mobileclip_s2')"
```

### Cách 2: Thủ công

**YOLOE Model** (70.8MB):
- Link: [Ultralytics Assets](https://github.com/ultralytics/assets/releases/)
- Đặt vào: `models/yoloe-11l-seg-pf.pt`

**MobileCLIP Model** (380MB):
- Link: [Hugging Face](https://huggingface.co/apple/MobileCLIP-S2-OpenCLIP)
- Đặt vào: `models/mobileclip_s2/mobileclip_s2.pt`

### Kiểm tra

```powershell
Test-Path models\yoloe-11l-seg-pf.pt              # → True
Test-Path models\mobileclip_s2\mobileclip_s2.pt  # → True
```

---

## 🚀 Sử Dụng

### Chạy pipeline

```powershell
.\.venv11\Scripts\Activate.ps1
python main.py --image data/images/image_01.jpg
```

### Kết quả mẫu

```
=== Food Detection Pipeline ===
[Loading] Models loaded
[Loading] Reference embeddings: 5 classes

[Processing] data/images/image_01.jpg
[YOLOE] Detected 4 objects → After filter: 3 objects
[MobileCLIP] Classification complete
[Saved] outputs/pipeline/image_01_result.jpg

Processing Time: 1.85s
```

**Output:**
- Bounding boxes màu theo class
- Label + confidence score
- Tự động mở cửa sổ xem kết quả

### Tùy chỉnh

```powershell
# Điều chỉnh confidence threshold
python main.py --image data/images/image_01.jpg --conf 0.3  # Phát hiện nhiều hơn
python main.py --image data/images/image_01.jpg --conf 0.7  # Chỉ phát hiện chắc chắn

# Custom output path
python main.py --image data/images/image_01.jpg --output results/my_result.jpg

# Batch processing
Get-ChildItem data\images\*.jpg | ForEach-Object {
    python main.py --image $_.FullName
}
```

---

## 📁 Cấu Trúc Project

```
ObjectDetection/
├── main.py                    # Entry point chính
├── requirements.txt           # Dependencies
│
├── models/                    # Models (KHÔNG push lên git)
│   ├── yoloe-11l-seg-pf.pt   # YOLOE (70.8MB) - phải tải riêng
│   └── mobileclip_s2/        # MobileCLIP (380MB) - phải tải riêng
│
├── src/                       # Source code
│   ├── yoloe_food.py         # Detector (YOLOE wrapper + filtering)
│   ├── embed.py              # MobileCLIP embeddings (512-dim vectors)
│   ├── visualize.py          # Vẽ bounding boxes + màu sắc
│   ├── match.py              # Similarity matching
│   ├── crop_food.py          # Crop detected regions
│   └── build_db.py           # Build embedding database
│
├── data/
│   ├── images/               # Ảnh input
│   ├── labels/               # YOLO labels (tham khảo)
│   └── ref_images/           # Ảnh tham chiếu (5 classes)
│       ├── coconut/
│       ├── cua/
│       ├── macaron/
│       ├── meden/
│       └── melon/
│
└── outputs/
    └── pipeline/             # Kết quả xử lý
```

---

## 🎨 Classes

5 classes dựa trên reference images trong `data/ref_images/`:

| Class | Mô tả | Số ảnh tham chiếu | Màu |
|-------|-------|-------------------|-----|
| `coconut` | Dừa | 6 | RGB(0, 100, 255) |
| `cua` | Cua/hải sản | 6 | RGB(255, 100, 0) |
| `macaron` | Macaron | 6 | RGB(100, 255, 0) |
| `meden` | Meden | 6 | RGB(255, 0, 100) |
| `melon` | Dưa | 6 | RGB(200, 200, 0) |

### Thêm class mới

1. Tạo thư mục: `data/ref_images/<tên_class>/`
2. Thêm 5-10 ảnh tham chiếu
3. Chạy lại pipeline → màu sẽ tự động tạo

---

## 🔧 Chi Tiết Kỹ Thuật

### Pipeline

1. **YOLOE Detection** → Phát hiện objects + segmentation masks
2. **Smart Filtering** → Loại bỏ khung trùng (>95% overlap), boxes quá lớn (>70%)
3. **MobileCLIP Embedding** → Tạo vector 512-dim cho mỗi object
4. **Classification** → So sánh với reference embeddings (cosine similarity)
5. **Visualization** → Vẽ boxes màu + labels

### Cấu hình

**Detection:**
- Confidence threshold: 0.5 (default)
- Overlap removal: >95%
- Large box filter: >70% image area
- Container classes: Tự động loại bỏ (table, tray, board...)

**Classification:**
- Embedding: 512 dimensions
- Similarity: Cosine similarity
- Threshold: Auto (highest average)

### Performance

- ~2s/ảnh (với 5 objects)
- Detection: 50%, Embedding: 47%, Visualization: 3%

---

## 🐛 Troubleshooting

**Models không tải được:**
```powershell
# Kiểm tra đường dẫn
Test-Path models\yoloe-11l-seg-pf.pt
Test-Path models\mobileclip_s2\mobileclip_s2.pt

# Xem kích thước file
Get-Item models\yoloe-11l-seg-pf.pt | Select-Object Length
```

**Không phát hiện được objects:**
- Giảm `--conf` xuống 0.3
- Kiểm tra ảnh input (JPG/PNG, độ phân giải hợp lý)

**Phân loại sai:**
- Thêm ảnh reference cho class đó (5-10 ảnh)
- Đảm bảo ảnh reference đa dạng (góc nhìn, ánh sáng)

**Import error:**
```powershell
pip install git+https://github.com/apple/ml-mobileclip.git
```

---

## 🤝 Contributing

Pull requests welcome! 

1. Fork repo
2. Tạo branch (`git checkout -b feature/NewFeature`)
3. Commit (`git commit -m 'Add NewFeature'`)
4. Push (`git push origin feature/NewFeature`)
5. Tạo Pull Request

---

## 📄 Credits

- [YOLOE](https://github.com/ultralytics/ultralytics) - Ultralytics
- [MobileCLIP](https://github.com/apple/ml-mobileclip) - Apple ML Research

---

**⭐ Nếu project hữu ích, hãy cho 1 star nhé!**

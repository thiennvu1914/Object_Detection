# Food Detection API

FastAPI-based REST API for food detection and classification.

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
pip install fastapi uvicorn python-multipart
```

### 2. Run API Server

```powershell
python run_api.py
```

Server will start at: **http://localhost:8000**

### 3. Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📡 API Endpoints

### `POST /api/v1/detect`

Detect and classify food items in an uploaded image.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Parameters:
  - `file`: Image file (JPG, PNG)
  - `confidence`: Detection confidence threshold (0.0-1.0, default: 0.5)

**Response:**
```json
{
  "success": true,
  "data": {
    "detections": [
      {
        "bbox": [100, 150, 300, 350],
        "class": "coconut",
        "similarity": 0.89,
        "confidence": 0.75,
        "index": 0
      }
    ],
    "count": 1,
    "processing_time": 1.85,
    "classes": ["coconut"],
    "image_shape": [480, 640, 3]
  }
}
```

**Example:**

```powershell
# Using curl
curl -X POST "http://localhost:8000/api/v1/detect?confidence=0.5" \
  -F "file=@path/to/image.jpg"

# Using Python requests
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/detect",
        files={"file": f},
        params={"confidence": 0.5}
    )
print(response.json())
```

---

### `GET /api/v1/classes`

Get list of available food classes.

**Response:**
```json
{
  "success": true,
  "data": {
    "classes": ["coconut", "cua", "macaron", "meden", "melon"],
    "count": 5
  }
}
```

**Example:**

```powershell
curl http://localhost:8000/api/v1/classes
```

---

### `POST /api/v1/detect-batch`

Process multiple images in one request.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Parameters:
  - `files`: List of image files (max 10)
  - `confidence`: Detection confidence threshold

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "filename": "image1.jpg",
        "success": true,
        "detections": [...],
        "count": 3,
        "processing_time": 1.5,
        "classes": ["coconut", "melon"]
      }
    ],
    "total": 1,
    "successful": 1
  }
}
```

---

## 🔌 Integration Examples

### Python Client

```python
import requests
from pathlib import Path

class FoodDetectionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def detect(self, image_path, confidence=0.5):
        """Detect food items in image"""
        with open(image_path, "rb") as f:
            response = requests.post(
                f"{self.base_url}/api/v1/detect",
                files={"file": f},
                params={"confidence": confidence}
            )
        return response.json()
    
    def get_classes(self):
        """Get available classes"""
        response = requests.get(f"{self.base_url}/api/v1/classes")
        return response.json()

# Usage
client = FoodDetectionClient()
result = client.detect("image.jpg", confidence=0.6)
print(f"Found {result['data']['count']} items")
for det in result['data']['detections']:
    print(f"  - {det['class']}: {det['similarity']:.2f}")
```

### JavaScript/Node.js Client

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function detectFood(imagePath, confidence = 0.5) {
    const form = new FormData();
    form.append('file', fs.createReadStream(imagePath));
    
    const response = await axios.post(
        `http://localhost:8000/api/v1/detect?confidence=${confidence}`,
        form,
        { headers: form.getHeaders() }
    );
    
    return response.data;
}

// Usage
detectFood('image.jpg', 0.6)
    .then(result => {
        console.log(`Found ${result.data.count} items`);
        result.data.detections.forEach(det => {
            console.log(`  - ${det.class}: ${det.similarity.toFixed(2)}`);
        });
    });
```

### cURL Examples

```bash
# Basic detection
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "file=@image.jpg"

# With custom confidence
curl -X POST "http://localhost:8000/api/v1/detect?confidence=0.7" \
  -F "file=@image.jpg"

# Get classes
curl http://localhost:8000/api/v1/classes

# Batch processing
curl -X POST "http://localhost:8000/api/v1/detect-batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Server settings
export API_HOST="0.0.0.0"
export API_PORT=8000

# Model paths
export YOLOE_MODEL="models/yoloe-11l-seg-pf.pt"
export MOBILECLIP_MODEL="models/mobileclip_s2"
export REF_IMAGES_DIR="data/ref_images"
```

### Production Deployment

```powershell
# Install production server
pip install gunicorn

# Run with Gunicorn
gunicorn food_detection.api.app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## 📊 Performance

- **Single image**: ~2s per request
- **Batch (10 images)**: ~15s total
- **Concurrent requests**: Supported via async/await

---

## 🔒 Security Notes

For production:
1. Configure CORS properly in `api/app.py`
2. Add authentication middleware
3. Implement rate limiting
4. Use HTTPS
5. Validate file uploads (size, type)

---

## 📝 License

MIT License

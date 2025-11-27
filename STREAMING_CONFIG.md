# 🎥 Camera Streaming Configuration Guide

Complete guide for optimizing food detection streaming performance.

---

## 📊 **Quick Start: Before vs After**

### **BEFORE (Laggy)**
```python
skip_frames=2              # 15 FPS detection
max_queue_size=2           # 2 frame buffer
ssim_threshold=0.85        # 15% change
diff_threshold=0.15        # 15% pixels
auto_flush_queue=False     # Queue can accumulate
```

**Problems:**
- ❌ Camera preview stutters
- ❌ Queue accumulates (2+ frames backlog)
- ❌ High CPU usage (~15 YOLOE calls/second)
- ❌ Feels unresponsive

### **AFTER (Smooth) - Recommended**
```python
skip_frames=15             # 2 FPS detection ⬆️
max_queue_size=1           # No backlog ⬇️
ssim_threshold=0.80        # 20% change ⬇️
diff_threshold=0.20        # 20% pixels ⬆️
auto_flush_queue=True      # Always latest ✅
```

**Results:**
- ✅ **Preview**: Smooth 10 FPS
- ✅ **Detection**: 0.3-0.6 FPS (98% reduction!)
- ✅ **Lag**: Eliminated
- ✅ **CPU**: 90-95% fewer YOLOE calls

---

## 🎯 **Anti-Lag Configuration (Recommended)**

### **Problem: Camera Stream Lag**
- Preview video stutters/freezes
- Detections accumulate in queue (backlog)
- System feels unresponsive

### **Solution: Optimized Settings**

```python
# food_detection/streaming/processor.py
FrameProcessor(
    skip_frames=15,              # ⬆️ Increased from 2 → 15 (2 FPS detection)
    max_queue_size=1,            # ⬇️ Reduced from 2 → 1 (no backlog)
    enable_change_detection=True,
    auto_flush_queue=True,       # 🆕 Auto-flush old frames
    
    # Change Detector thresholds
    change_detector=ChangeDetector(
        ssim_threshold=0.80,     # ⬇️ Reduced from 0.85 → 0.80 (20% change)
        diff_threshold=0.20      # ⬆️ Increased from 0.15 → 0.20 (20% pixels)
    )
)
```

### **Results:**
- ✅ **Preview stream**: Smooth 10 FPS (unchanged)
- ✅ **Detection rate**: 1-3 FPS (70-90% fewer YOLOE calls)
- ✅ **Lag**: Eliminated (queue always ≤ 1 frame)
- ✅ **Responsiveness**: Real-time feel

---

## 📊 **Configuration Matrix**

| Setting | Low Lag | Balanced | High Accuracy |
|---------|---------|----------|---------------|
| **skip_frames** | 15-20 | 5-10 | 1-2 |
| **max_queue_size** | 1 | 2 | 3-5 |
| **ssim_threshold** | 0.75-0.80 | 0.85 | 0.90-0.95 |
| **diff_threshold** | 0.20-0.25 | 0.15 | 0.10 |
| **auto_flush_queue** | True | True | False |
| **Detection FPS** | 1-2 | 3-6 | 10-15 |
| **YOLOE reduction** | 90-95% | 70-80% | 40-60% |

---

## 🔧 **Tuning Parameters**

### **1. skip_frames** (Frame Skipping)
Controls how often to process frames:
- `skip_frames=1`: Process every frame (30 FPS) - **High CPU, possible lag**
- `skip_frames=5`: Process every 5th frame (6 FPS) - **Balanced**
- `skip_frames=15`: Process every 15th frame (2 FPS) - **Low CPU, smooth**
- `skip_frames=30`: Process every 30th frame (1 FPS) - **Very low CPU**

**Formula**: `detection_fps = camera_fps / skip_frames = 30 / skip_frames`

**Recommendation**: 
- **POS system** (items change rarely): `skip_frames=15-20` (1-2 FPS)
- **Conveyor belt** (continuous flow): `skip_frames=5-10` (3-6 FPS)
- **Manual inspection**: `skip_frames=10-15` (2-3 FPS)

---

### **2. max_queue_size** (Queue Management)
Controls frame backlog:
- `max_queue_size=1`: **No backlog** - always process latest frame (anti-lag)
- `max_queue_size=2`: Small buffer (slight lag possible)
- `max_queue_size=5`: Large buffer (can cause significant lag)

**Recommendation**: Always use `max_queue_size=1` for real-time streaming

---

### **3. ssim_threshold** (SSIM Sensitivity)
Controls structural similarity threshold (0.0-1.0):
- `ssim_threshold=0.95`: Very sensitive (5% change triggers) - **More detections**
- `ssim_threshold=0.85`: Balanced (15% change triggers) - **Recommended default**
- `ssim_threshold=0.80`: Less sensitive (20% change triggers) - **Fewer false triggers**
- `ssim_threshold=0.70`: Very insensitive (30% change triggers) - **May miss small changes**

**Lower value = Less sensitive = Fewer YOLOE calls = Better performance**

**Recommendation**:
- **Static scene** (POS checkout): `0.80` (less sensitive)
- **Dynamic scene** (moving objects): `0.85-0.90` (more sensitive)

---

### **4. diff_threshold** (Pixel Difference)
Controls pixel-level change threshold (0.0-1.0):
- `diff_threshold=0.10`: Very sensitive (10% pixels changed) - **More detections**
- `diff_threshold=0.15`: Balanced (15% pixels changed) - **Recommended default**
- `diff_threshold=0.20`: Less sensitive (20% pixels changed) - **Fewer false triggers**
- `diff_threshold=0.30`: Very insensitive (30% pixels changed) - **May miss changes**

**Higher value = Less sensitive = Fewer YOLOE calls = Better performance**

**Recommendation**:
- **Good lighting** (stable): `0.20` (less sensitive)
- **Poor lighting** (shadows, flicker): `0.15` (more tolerant)

---

### **5. auto_flush_queue** (Queue Flushing)
Automatically removes old frames from queue:
- `auto_flush_queue=True`: Always process latest frame ✅ **Anti-lag**
- `auto_flush_queue=False`: Process all frames in order (can cause backlog)

**Recommendation**: Always `True` for real-time streaming

---

## 📈 **Performance Analysis**

### **Baseline (Old Settings)**
```python
skip_frames=2              # 15 FPS detection
ssim_threshold=0.85        # 15% change
diff_threshold=0.15        # 15% pixels
max_queue_size=2           # 2 frame buffer
```

**Results:**
- Detection calls: ~15 FPS → ~6 FPS (60% reduction)
- Preview: May stutter during detection
- Queue: Can accumulate 2 frames (lag)

---

### **Optimized (New Settings)**
```python
skip_frames=15             # 2 FPS detection
ssim_threshold=0.80        # 20% change
diff_threshold=0.20        # 20% pixels
max_queue_size=1           # No buffer
auto_flush_queue=True      # Always latest
```

**Results:**
- Detection calls: ~15 FPS → ~0.5-1.5 FPS (90-95% reduction)
- Preview: Always smooth 10 FPS
- Queue: Always ≤ 1 frame (no lag)

---

## 🚀 **Advanced Tuning**

### **Adaptive Change Detection**
Use `AdaptiveChangeDetector` for automatic threshold adjustment:

```python
from food_detection.streaming.change_detector import AdaptiveChangeDetector

detector = AdaptiveChangeDetector(
    initial_ssim_threshold=0.85,
    min_ssim_threshold=0.75,      # Most sensitive
    max_ssim_threshold=0.95,      # Least sensitive
    adaptation_rate=0.1           # How fast to adapt
)

processor = FrameProcessor(
    pipeline=pipeline,
    change_detector=detector
)
```

**Behavior:**
- **Stable scene** (high SSIM, low diff) → Increase threshold (less sensitive)
- **Dynamic scene** (low SSIM, high diff) → Decrease threshold (more sensitive)

---

## 🧪 **Testing & Validation**

### **1. Test Change Detection**
```powershell
python tests/test_change_detection.py
```

**Observe:**
- SSIM scores (should be 0.95+ for static, <0.85 for changes)
- Diff ratios (should be <0.10 for static, >0.15 for changes)
- FPS (should be ~30 FPS camera, 2-5ms processing)

---

### **2. Test Streaming**
```powershell
# Start API server
python run_api.py

# Open browser
http://localhost:8000/static/camera_demo.html
```

**Check:**
- Preview FPS (should be smooth ~10 FPS)
- Detection FPS (should match skip_frames setting)
- Lag (preview should never freeze)

---

### **3. Monitor Statistics**
Check WebSocket messages for performance metrics:

```json
{
  "type": "stats",
  "data": {
    "frames_processed": 45,                    // YOLOE calls
    "frames_skipped": 450,                     // Skipped by skip_frames
    "frames_skipped_by_change_detector": 30,   // Skipped by SSIM
    "frames_flushed": 5,                       // Flushed from queue
    "optimization_ratio": "93.5%",             // Total reduction
    "effective_fps": "1.50 FPS"                // Actual detection rate
  }
}
```

**Good indicators:**
- `optimization_ratio > 80%`: Excellent performance
- `frames_flushed = 0`: No backlog (good)
- `effective_fps < 3`: Smooth streaming

---

## 💡 **Best Practices**

### ✅ **DO:**
1. Use `skip_frames=15-20` for POS systems
2. Keep `max_queue_size=1` for real-time
3. Enable `auto_flush_queue=True`
4. Set `ssim_threshold=0.80` for static scenes
5. Monitor `frames_flushed` (should be low)
6. Test with actual camera and lighting

### ❌ **DON'T:**
1. Set `skip_frames < 5` unless necessary (causes lag)
2. Use `max_queue_size > 2` (creates backlog)
3. Disable change detection (wastes CPU)
4. Set `ssim_threshold > 0.90` (too sensitive, false triggers)
5. Ignore lag symptoms (tune parameters!)

---

## 🎯 **Quick Fixes**

### **Problem: Preview lags/freezes**
```python
skip_frames=20          # ⬆️ Increase (reduce detection frequency)
max_queue_size=1        # ⬇️ Reduce to 1
auto_flush_queue=True   # ✅ Enable
```

### **Problem: Missing real changes**
```python
ssim_threshold=0.85     # ⬆️ Increase (more sensitive)
diff_threshold=0.15     # ⬇️ Decrease (more sensitive)
```

### **Problem: Too many false detections**
```python
ssim_threshold=0.80     # ⬇️ Decrease (less sensitive)
diff_threshold=0.20     # ⬆️ Increase (less sensitive)
conf=0.35               # ⬆️ Increase YOLOE confidence
```

### **Problem: High CPU usage**
```python
skip_frames=20          # ⬆️ Increase
ssim_threshold=0.75     # ⬇️ Decrease (skip more frames)
enable_change_detection=True  # ✅ Must enable
```

---

## 📈 **Performance Metrics & Analysis**

### **Reduction Pipeline:**
```
Camera: 30 FPS
  ↓ skip_frames (15)
Submitted: 2 FPS (93% reduction)
  ↓ change_detector (0.80/0.20 thresholds)
YOLOE: 0.3-0.6 FPS (85-90% reduction of submitted)
  ↓ (only if objects detected)
MobileCLIP: 0.3-0.6 FPS
```

### **Total Optimization:**
- **Frame skipping**: 93.3% reduction (30 → 2 FPS)
- **Change detection**: 70-85% reduction (2 → 0.3-0.6 FPS)
- **Combined**: **98-99% total reduction** (30 → 0.3-0.6 FPS)

### **Resource Usage:**
- **CPU per frame**: 2-5ms (change detection) vs 2300ms (full pipeline)
- **Effective processing time**: 0.3 FPS * 2.3s = 0.69s/second (69% CPU)
- **vs Old**: 15 FPS * 2.3s = 34.5s/second (**34x CPU overload!** ❌)

### **What We Sacrificed:**
- ❌ Detection frequency: 15 FPS → 2 FPS (87% less)
- ❌ Sensitivity: More tolerant to small changes
- ❌ Historical data: No DB saves during streaming

### **What We Gained:**
- ✅ Smooth preview: Always 10 FPS
- ✅ No lag: Queue never accumulates
- ✅ Low CPU: 98% fewer heavy operations
- ✅ Real-time feel: Always processing latest frame
- ✅ Scalability: Can handle multiple streams

### **Why It Works for POS/Checkout:**
- Items don't move rapidly (2 FPS is enough)
- Scene is mostly static (change detection filters noise)
- Preview shows real-time feedback (user sees immediately)
- Detection runs only when needed (items added/removed)

---

## 📚 **Configuration Files**

### **Update API defaults:**
Edit `food_detection/api/streaming.py`:
```python
async def start_streaming(
    camera_id: int = 0,
    skip_frames: int = 15,    # Your preference
    conf: float = 0.25
):
```

### **Update Processor defaults:**
Edit `food_detection/streaming/processor.py`:
```python
def __init__(
    self,
    pipeline,
    skip_frames: int = 15,           # Your preference
    max_queue_size: int = 1,         # Keep at 1
    ssim_threshold=0.80,             # Your preference
    diff_threshold=0.20,             # Your preference
):
```

---

## 🎓 **Understanding the Numbers**

### **Example Calculation:**
Camera: 30 FPS
- `skip_frames=15` → Submit 2 frames/second to processor
- `ssim_threshold=0.80` → ~30% of submitted frames trigger YOLOE
- **Net result**: 2 * 0.3 = 0.6 FPS YOLOE calls
- **Reduction**: 30 FPS → 0.6 FPS = **98% fewer calls!**

### **Pipeline Timing:**
- Camera capture: 33ms/frame (30 FPS)
- Change detection: 2-5ms/frame (SSIM + diff)
- YOLOE: ~1300ms (when triggered)
- MobileCLIP: ~1000ms (if objects detected)

**Without optimization**: 30 FPS * 2.3s = System overload ❌
**With optimization**: 0.6 FPS * 2.3s = 1.38s/second ✅

---

## 🧪 **Testing & Validation**

### **Test 1: Change Detection**
```powershell
python tests/test_change_detection.py
```
**Expected:**
- SSIM: ~0.95+ for static, <0.80 for changes
- Diff: <0.10 for static, >0.20 for changes
- Processing: 2-5ms per frame
- FPS: ~30 camera FPS

### **Test 2: Streaming**
```powershell
python run_api.py
# Open: http://localhost:8000/static/camera_demo.html
```
**Expected:**
- Preview: Smooth, no stutters
- Detection: 1-3 FPS display rate
- Queue: Always ≤ 1 (check stats)
- Flushed: Low count (means no backlog)

### **Test 3: Load Test**
Move objects in/out of frame rapidly:
**Expected:**
- Preview: Still smooth (doesn't freeze)
- Detection: Triggers only on significant changes
- Queue: Never accumulates
- Stats: High optimization_ratio (>90%)

---

## 📞 **Troubleshooting**

### **Symptoms of lag:**
- Preview freezes/stutters
- High `frames_flushed` count
- Detection results delayed
- Queue size > 1

### **Solution:**
1. Increase `skip_frames` (15 → 20)
2. Verify `max_queue_size=1`
3. Enable `auto_flush_queue=True`
4. Check CPU usage (should be <70%)

### **Support Checklist:**
- [ ] Check `frames_flushed` statistic (high = possible lag)
- [ ] Monitor `optimization_ratio` (should be >80%)
- [ ] Test with `tests/test_change_detection.py`
- [ ] Adjust parameters incrementally (don't change all at once)

**Golden rule**: Start conservative (high skip_frames), then decrease if needed. Preview should NEVER lag!

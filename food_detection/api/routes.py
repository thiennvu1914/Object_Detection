"""
API Routes
==========
REST API endpoints for food detection.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import tempfile
import time
from typing import Optional

from ..core.pipeline import FoodDetectionPipeline


# Create router
router = APIRouter(prefix="/api/v1", tags=["food-detection"])

# Initialize pipeline (lazy loading)
_pipeline = None


def get_pipeline():
    """Get or initialize pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = FoodDetectionPipeline()
    return _pipeline


@router.post("/detect")
async def detect_food(
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.0, le=1.0, description="Detection confidence threshold")
):
    """
    Detect and classify food items in an uploaded image.
    
    Args:
        file: Image file (JPG, PNG)
        confidence: Detection confidence threshold (0.0-1.0)
        
    Returns:
        JSON with detection results:
        - detections: List of detected food items with bbox, class, score
        - count: Number of detected items
        - processing_time: Processing time in seconds
        - classes: List of detected class names
    """
    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Additional validation: check file extension if content_type is None
    if not file.content_type:
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"File must be an image. Allowed: {allowed_extensions}")
    
    # Log input
    print(f"\n[API] INPUT: image={file.filename}, confidence={confidence}")
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        # Process image
        pipeline = get_pipeline()
        results = pipeline.process_image(tmp_path, conf=confidence)
        
        # Extract class names
        classes = [det['class'] for det in results['detections']]
        
        # Log output
        if len(results['detections']) > 0:
            print(f"[API] OUTPUT: status=OK (detected {len(results['detections'])} items: {classes})")
        else:
            print(f"[API] OUTPUT: status=FAILED (no items detected)")
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "detections": results['detections'],
                "count": len(results['detections']),
                "processing_time": results['processing_time'],
                "classes": list(set(classes)),
                "image_shape": results['image_shape']
            }
        })
        
    except Exception as e:
        print(f"[API] OUTPUT: status=FAILED (error: {str(e)})")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Cleanup temporary file
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/classes")
async def get_classes():
    """
    Get list of available food classes.
    
    Returns:
        JSON with list of available classes and count
    """
    try:
        pipeline = get_pipeline()
        classes = pipeline.get_available_classes()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "classes": classes,
                "count": len(classes)
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/detect-batch")
async def detect_food_batch(
    files: list[UploadFile] = File(...),
    confidence: float = Query(0.5, ge=0.0, le=1.0)
):
    """
    Detect and classify food items in multiple images.
    
    Args:
        files: List of image files
        confidence: Detection confidence threshold
        
    Returns:
        JSON with results for each image
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    # Log input
    file_names = [f.filename for f in files]
    print(f"\n[API] INPUT: batch_size={len(files)}, images={file_names}, confidence={confidence}")
    
    results = []
    pipeline = get_pipeline()
    
    for file in files:
        # Validate file type
        is_valid = False
        if file.content_type:
            is_valid = file.content_type.startswith("image/")
        else:
            # Fallback to extension check
            allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            file_ext = Path(file.filename).suffix.lower()
            is_valid = file_ext in allowed_extensions
        
        if not is_valid:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": "Not an image file"
            })
            continue
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        try:
            # Process image
            result = pipeline.process_image(tmp_path, conf=confidence)
            classes = [det['class'] for det in result['detections']]
            
            results.append({
                "filename": file.filename,
                "success": True,
                "detections": result['detections'],
                "count": len(result['detections']),
                "processing_time": result['processing_time'],
                "classes": list(set(classes))
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
        
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    # Log output
    successful = sum(1 for r in results if r.get("success", False))
    statuses = []
    for r in results:
        if r.get("success") and r.get("count", 0) > 0:
            statuses.append(f"{r['filename']}=OK")
        else:
            statuses.append(f"{r['filename']}=FAILED")
    
    overall_status = "OK" if successful == len(files) and all("=OK" in s for s in statuses) else "FAILED"
    print(f"[API] OUTPUT: overall_status={overall_status}, successful={successful}/{len(files)}")
    print(f"[API]    Details: {', '.join(statuses)}")
    
    return JSONResponse(content={
        "success": True,
        "data": {
            "results": results,
            "total": len(files),
            "successful": sum(1 for r in results if r.get("success", False))
        }
    })

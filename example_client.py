"""
Example API Client
==================
Demonstrates how to use the Food Detection API.
"""
import requests
from pathlib import Path
from typing import Optional


class FoodDetectionClient:
    """Client for Food Detection API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.api_base = f"{self.base_url}/api/v1"
    
    def detect(self, image_path: str, confidence: float = 0.5) -> dict:
        """
        Detect food items in an image.
        
        Args:
            image_path: Path to image file
            confidence: Detection confidence threshold (0.0-1.0)
            
        Returns:
            API response dict with detection results
        """
        print(f"\nINPUT: image={Path(image_path).name}")
        
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.api_base}/detect",
                files={'file': f},
                params={'confidence': confidence}
            )
        response.raise_for_status()
        result = response.json()
        
        # Check if any items detected
        if result.get('success') and result['data'].get('count', 0) > 0:
            status = "OK"
            detected_classes = result['data'].get('classes', [])
            print(f"OUTPUT: status={status} (detected={detected_classes})")
        else:
            status = "FAILED"
            print(f"OUTPUT: status={status} (no items detected)")
        
        return result
    
    def detect_batch(self, image_paths: list, confidence: float = 0.5) -> dict:
        """
        Detect food items in multiple images.
        
        Args:
            image_paths: List of image file paths (max 10)
            confidence: Detection confidence threshold
            
        Returns:
            API response dict with batch results
        """
        if len(image_paths) > 10:
            raise ValueError("Maximum 10 images per batch")
        
        # Log input
        image_names = [Path(p).name for p in image_paths]
        print(f"\nINPUT: images={image_names}")
        
        files = [('files', open(path, 'rb')) for path in image_paths]
        try:
            response = requests.post(
                f"{self.api_base}/detect-batch",
                files=files,
                params={'confidence': confidence}
            )
            response.raise_for_status()
            result = response.json()
            
            # Check results
            if result.get('success'):
                statuses = []
                for i, img_result in enumerate(result['data']['results']):
                    if img_result.get('success') and img_result.get('count', 0) > 0:
                        statuses.append(f"{image_names[i]}=OK")
                    else:
                        statuses.append(f"{image_names[i]}=FAILED")
                
                overall_status = "OK" if all("=OK" in s for s in statuses) else "FAILED"
                print(f"OUTPUT: overall_status={overall_status}")
                print(f"   Details: {', '.join(statuses)}")
            else:
                print(f"OUTPUT: status=FAILED")
            
            return result
        finally:
            for _, f in files:
                f.close()
    
    def get_classes(self) -> dict:
        """
        Get list of available food classes.
        
        Returns:
            API response dict with class list
        """
        response = requests.get(f"{self.api_base}/classes")
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> dict:
        """
        Check API server health.
        
        Returns:
            Health status dict
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


def main():
    """Example usage"""
    # Initialize client
    client = FoodDetectionClient("http://localhost:8000")
    
    # Health check
    print("=== Health Check ===")
    health = client.health_check()
    print(f"Status: {health['status']}")
    
    # Get available classes
    print("\n=== Available Classes ===")
    classes_response = client.get_classes()
    classes = classes_response['data']['classes']
    print(f"Classes: {', '.join(classes)}")
    print(f"Total: {len(classes)}")
    
    # Detect in single image
    print("\n=== Single Image Detection ===")
    image_path = "data/images/image_01.jpg"
    
    if Path(image_path).exists():
        result = client.detect(image_path, confidence=0.5)
        
        if result['success']:
            data = result['data']
            print(f"Processed in {data['processing_time']:.2f}s")
            print(f"Found {data['count']} items:")
            
            for det in data['detections']:
                print(f"  - {det['class']}: similarity={det['similarity']:.2f}, conf={det['confidence']:.2f}")
                print(f"    bbox: {det['bbox']}")
        else:
            print("Detection failed")
    else:
        print(f"Image not found: {image_path}")
    
    # Batch detection example
    print("\n=== Batch Detection ===")
    batch_paths = [
        "data/images/image_01.jpg",
        "data/images/image_02.jpg",
    ]
    
    # Filter existing files
    existing_paths = [p for p in batch_paths if Path(p).exists()]
    
    if existing_paths:
        batch_result = client.detect_batch(existing_paths, confidence=0.5)
        
        if batch_result['success']:
            data = batch_result['data']
            print(f"Processed {data['total']} images, {data['successful']} successful")
            
            for result in data['results']:
                if result['success']:
                    print(f"\n  {result['filename']}: {result['count']} items")
                    for det in result['detections']:
                        print(f"    - {det['class']}: {det['similarity']:.2f}")
                else:
                    print(f"\n  {result['filename']}: Error - {result['error']}")
    else:
        print("No valid image files found")


if __name__ == "__main__":
    main()

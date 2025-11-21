"""
Quick API Test Script
Run this while API server is running (python run_api.py)
"""
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_get_classes():
    """Test get classes endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Get Available Classes")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/api/v1/classes")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Classes: {data['data']['classes']}")
    print(f"Count: {data['data']['count']}")
    return response.status_code == 200

def test_detect_single():
    """Test single image detection"""
    print("\n" + "="*60)
    print("TEST 3: Detect Single Image")
    print("="*60)
    
    image_path = Path("data/images/image_01.jpg")
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    print(f"Uploading: {image_path}")
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        params = {"confidence": 0.5}
        response = requests.post(
            f"{BASE_URL}/api/v1/detect",
            files=files,
            params=params
        )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Success: {data['success']}")
        print(f"✓ Detections: {data['data']['count']}")
        print(f"✓ Processing Time: {data['data']['processing_time']:.2f}s")
        print(f"✓ Classes Found: {data['data']['classes']}")
        
        print("\nDetection Details:")
        for i, det in enumerate(data['data']['detections'], 1):
            print(f"  {i}. {det['class']}: similarity={det['similarity']:.3f}, conf={det['confidence']:.3f}")
        
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False

def main():
    print("="*60)
    print("🧪 FOOD DETECTION API - TEST SUITE")
    print("="*60)
    print("Make sure API server is running: python run_api.py")
    print("API should be at: http://localhost:8000")
    
    # Run tests
    results = []
    
    try:
        results.append(("Health Check", test_health()))
        results.append(("Get Classes", test_get_classes()))
        results.append(("Single Detection", test_detect_single()))
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API server!")
        print("Please start server first: python run_api.py")
        return
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed!")

if __name__ == "__main__":
    main()

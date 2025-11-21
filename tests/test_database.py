"""
Test database integration
"""
from food_detection.database import DatabaseManager
from food_detection.core.pipeline import FoodDetectionPipeline

# Test database
print("="*60)
print("Testing Database Integration")
print("="*60)

# Initialize pipeline (will create database and cache embeddings)
print("\n1. Initializing pipeline...")
pipeline = FoodDetectionPipeline(use_cache=True)

# Check database embeddings
print("\n2. Checking cached embeddings...")
counts = pipeline.db.get_embeddings_count()
print(f"   Cached embeddings: {counts}")

# Process an image
print("\n3. Processing test image...")
result = pipeline.process_image("data/images/image_01.jpg", conf=0.5)
print(f"   Detected {len(result['detections'])} objects")

# Save to database
if len(result['detections']) > 0:
    session_id = pipeline.db.save_detection_session(
        image_filename="image_01.jpg",
        detections=result['detections']
    )
    print(f"   Saved to database (session_id={session_id})")

# Check detection history
print("\n4. Checking detection history...")
recent = pipeline.db.get_recent_sessions(limit=5)
print(f"   Total sessions: {len(recent)}")
for session in recent:
    print(f"   - {session['image_filename']}: {session['total_objects']} objects")
    print(f"     Classes: {session['detected_classes']}")

# Get statistics
print("\n5. Class statistics...")
stats = pipeline.db.get_class_statistics()
for stat in stats:
    print(f"   - {stat['class_name']}: {stat['total_detections']} detections, "
          f"avg similarity: {stat['avg_similarity']}")

print("\n✅ Database integration test completed!")

"""
API Server Entry Point
=======================
Run FastAPI server for food detection.

Usage:
    python run_api.py

Server will start at:
    - http://localhost:8000
    - Docs: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
"""
import uvicorn
from food_detection.api.app import app


if __name__ == "__main__":
    uvicorn.run(
        "food_detection.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )

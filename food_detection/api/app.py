"""
FastAPI Application
===================
REST API for food detection and classification.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .routes import router
from .streaming import router as streaming_router
from .predict_stream import router as predict_router
from .training_stream import router as training_router


# Create FastAPI app
app = FastAPI(
    title="Food Detection API",
    description="API for detecting and classifying food items using YOLOE and MobileCLIP",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for HTML demo)
static_path = Path(__file__).parent.parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Include routes
app.include_router(router)
app.include_router(streaming_router)
app.include_router(predict_router)
app.include_router(training_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Food Detection API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

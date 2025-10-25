import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ..services.image_service import ImageService
from ..services.video_service import VideoService
from ..core.config import settings
import logging

router = APIRouter(prefix="/api/v1/predict", tags=["prediction"])
logger = logging.getLogger(__name__)

image_service = ImageService()
video_service = VideoService()

@router.post("/image")
async def predict_image(file: UploadFile = File(...)):
    """API for image prediction"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(tuple(f".{ext}" for ext in settings.ALLOWED_IMAGE_EXTENSIONS)):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Invalid image format. Allowed: {settings.ALLOWED_IMAGE_EXTENSIONS}",
                    "data": None,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        result = await image_service.process_image(file)
        
        # Nếu service trả về error, return status code phù hợp
        if result.get("status") == "error":
            return JSONResponse(status_code=500, content=result)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Image prediction error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Internal server error: {str(e)}",
                "data": None,
                "timestamp": datetime.now().isoformat()
            }
        )

@router.post("/video")
async def predict_video(file: UploadFile = File(...)):
    """API for video prediction"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(tuple(f".{ext}" for ext in settings.ALLOWED_VIDEO_EXTENSIONS)):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Invalid video format. Allowed: {settings.ALLOWED_VIDEO_EXTENSIONS}",
                    "data": None,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        result = await video_service.process_video(file)
        
        if result.get("status") == "error":
            return JSONResponse(status_code=500, content=result)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Video prediction error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Internal server error: {str(e)}",
                "data": None,
                "timestamp": datetime.now().isoformat()
            }
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "success",
        "message": "Service is healthy",
        "data": {
            "image_model_loaded": image_service.model is not None,
            "video_model_loaded": video_service.model is not None
        },
        "timestamp": datetime.now().isoformat()
    }
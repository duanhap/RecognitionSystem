from datetime import datetime
import traceback 
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ..services.image_service import ImageService
from ..services.video_service import VideoService
from ..services.identity_verification_service import IdentityVerificationService
from ..core.config import settings
import logging

router = APIRouter(prefix="/api/v1/predict", tags=["prediction"])
logger = logging.getLogger(__name__)

image_service = ImageService()
video_service = VideoService()
identity_verification_service = IdentityVerificationService()  # Chỉ tạo instance

@router.on_event("startup")
async def startup_event():
    """Khởi tạo services khi app startup"""
    try:
        logger.info("🔄 Starting services initialization...")
        
        # Khởi tạo identity service
        logger.info("🔄 Initializing IdentityVerificationService...")
        await identity_verification_service.initialize()
        
        logger.info(f"✅ IdentityVerificationService initialized. Loaded {len(identity_verification_service.embedding_db) if identity_verification_service.embedding_db else 0} embeddings")
        logger.info("✅ All services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ FAILED to initialize services: {e}")
        logger.error(f"🔍 Stack trace: {traceback.format_exc()}")
        raise

@router.post("/combined")
async def predict_combined(
    file: UploadFile = File(...),
    file_type: str = "auto"  # auto, image, video
):
    """API for combined fake detection and identity verification"""
    try:
        # Kiểm tra service đã sẵn sàng chưa
        if identity_verification_service.embedding_db is None:
            await identity_verification_service.initialize()
        
        # Auto detect file type
        if file_type == "auto":
            ext = file.filename.split('.')[-1].lower()
            if ext in settings.ALLOWED_IMAGE_EXTENSIONS:
                file_type = "image"
            elif ext in settings.ALLOWED_VIDEO_EXTENSIONS:
                file_type = "video"
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": "Unsupported file format",
                        "data": None,
                        "timestamp": datetime.now().isoformat()
                    }
                )
        
        result = await identity_verification_service.process_combined(file, file_type)
        
        if result.get("status") == "error":
            return JSONResponse(status_code=500, content=result)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Combined prediction error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Internal server error: {str(e)}",
                "data": None,
                "timestamp": datetime.now().isoformat()
            }
        )
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
            "video_model_loaded": video_service.model is not None,
            "identity_service_ready": identity_verification_service.embedding_db is not None,
            "embedding_count": len(identity_verification_service.embedding_db) if identity_verification_service.embedding_db else 0
        },
        "timestamp": datetime.now().isoformat()
    }
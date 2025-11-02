import logging
from fastapi import FastAPI, logger
from fastapi.staticfiles import StaticFiles
from .core.model_loader import model_loader
from .core.config import settings

from app.routers.manage_user_router import manage_user_router
from app.routers.login_router import login_router
from app.routers.home_router import home_router
from app.routers.view_detail_user_router import view_detail_user_router
from app.routers.view_detail_sample_router import view_detail_sample_router
from app.routers.manage_sample_router import manage_sample_router
from app.routers.add_multiple_samples_router import add_multiple_samples_router
from app.routers.training_router import training_router
from fastapi.middleware.cors import CORSMiddleware
from app.routers.add_single_sample_router import add_single_sample_router
from app.routers.view_feedback_router import feedback_router
from app.routers.manage_test_result_router import manage_test_result_router
from app.routers.view_detail_result_router import view_detail_result_router
from app.routers.training_config_router import training_config_router
from .routers import  predict_router
from starlette.middleware.sessions import SessionMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
# Mount thư mục static
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory="app/dataset"), name="uploads")
# ✅ Thêm SessionMiddleware
app.add_middleware(SessionMiddleware, secret_key="my_super_secret_key")
app.include_router(login_router)
app.include_router(home_router)
app.include_router(manage_user_router)
app.include_router(manage_sample_router)
app.include_router(add_multiple_samples_router)
app.include_router(view_detail_sample_router)
app.include_router(add_single_sample_router)
app.include_router(view_detail_user_router)
app.include_router(feedback_router)
app.include_router(manage_test_result_router)
app.include_router(view_detail_result_router)
app.include_router(training_config_router)

app.include_router(predict_router.router)
app.include_router(training_router)


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("🚀 Starting Deepfake Detection API...")
    model_loader.load_models()
    
    # ✅ Load training config từ file (cách đơn giản)
    try:
        from app.routers.training_config_router import TrainingConfigRouter
        config_handler = TrainingConfigRouter()
        config_handler.load_config_from_file()
        logger.info("✓ Training config loaded from file")
    except Exception as e:
        logger.warning(f"Could not load training config: {e}")

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
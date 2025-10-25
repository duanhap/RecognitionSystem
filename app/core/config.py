import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings:
    # Model paths
    MODEL_SYSTEM_DIR = BASE_DIR / "app/modelsystem"
    IMAGE_MODEL_PATH = MODEL_SYSTEM_DIR / "image" / "20251025_152553_Xception" / "model_final.h5"
    VIDEO_MODEL_PATH = MODEL_SYSTEM_DIR / "video" / "20251025_171325_Xception_video" / "model_final.h5"
    
    # Upload directories
    UPLOAD_DIR = BASE_DIR / "app/uploads"
    IMAGE_UPLOAD_DIR = UPLOAD_DIR / "images"
    VIDEO_UPLOAD_DIR = UPLOAD_DIR / "videos"
    
    # Static directories
    STATIC_DIR = BASE_DIR / "app/static"
    HEATMAP_DIR = STATIC_DIR / "heatmaps"
    IMAGE_HEATMAP_DIR = HEATMAP_DIR / "images"
    VIDEO_HEATMAP_DIR = HEATMAP_DIR / "videos"
    
    # API settings
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}
    ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

settings = Settings()

# Create directories
for directory in [settings.IMAGE_UPLOAD_DIR, settings.VIDEO_UPLOAD_DIR, 
                  settings.IMAGE_HEATMAP_DIR, settings.VIDEO_HEATMAP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
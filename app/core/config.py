import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings:
    # Model paths
    MODEL_SYSTEM_DIR = BASE_DIR / "app/modelsystem"
    IMAGE_MODEL_PATH = MODEL_SYSTEM_DIR / "image" / "20251031_120748_continue_from_20251031_114053_Xception" / "model_final.h5"
    VIDEO_MODEL_PATH = MODEL_SYSTEM_DIR / "video" / "20251031_223619_continue_superdeep_20251031_215958_Xception_video" / "model_final.h5"
    IDENTITY_MODEL_PATH = MODEL_SYSTEM_DIR / "identity" / "20251109_145749_identity_verification" / "model_final.h5"
    
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

    # Thêm cấu hình cho video processing
    VIDEO_FRAME_INTERVAL = 10
    MAX_VIDEO_FRAMES = 30
    VIDEO_POOLING_STRATEGY = "confidence_weighted"  # mean, max, median, q75, confidence_weighted

        # Image Training Defaults
    IMAGE_TRAINING_CONFIG = {
        "depth_presets": {
            "normal": {"epochs": 10, "patience": 3, "lr": 1e-4, "batch_size": 32},
            "deep": {"epochs": 20, "patience": 5, "lr": 5e-5, "batch_size": 24},
            "superdeep": {"epochs": 30, "patience": 7, "lr": 1e-5, "batch_size": 16},
        },
        "default_depth": "normal",
        "default_n_samples": 1000,
        "default_sampling_mode": "random",  # "random" or "newest"
        "default_train_ratio": 0.8,
        "img_size": (299, 299),
        "default_dataset_root": "dataset/image"
    }
    
    # Video Training Defaults  
    VIDEO_TRAINING_CONFIG = {
        "depth_presets": {
            "normal": {"epochs": 10, "patience": 3, "lr": 1e-4, "batch_size": 32},
            "deep": {"epochs": 20, "patience": 5, "lr": 5e-5, "batch_size": 24},
            "superdeep": {"epochs": 30, "patience": 7, "lr": 1e-5, "batch_size": 16},
        },
        "default_depth": "normal",
        "default_n_samples": 500,
        "default_sampling_mode": "random",
        "default_train_ratio": 0.8,
        "default_pooling_strategy": "mean",  # "mean", "max", "median", "q75", "confidence_weighted"
        "default_frame_interval": 10,
        "default_max_frames_per_video": 50,
        "default_force_extract": False,
        "img_size": (299, 299),
        "default_dataset_root": "dataset/video"
    }
    IDENTITY_CONFIG = {
        "model_type": "facenet",  
        "embedding_dim": 512,
        "frame_interval": 10,
        "max_frames_per_video": 10,
        "video_pooling": "mean",  # "mean", "max", "median"
        "train_ratio": 0.8,
        "batch_size": 32,
        "epochs": 20,
        "learning_rate": 1e-4,
        "patience": 5,
        "default_dataset_root": "dataset2"
    }
     # Training Output
    TRAINING_OUTPUT_DIRS = {
        "image": BASE_DIR / "models/image",
        "video": BASE_DIR / "models/video",
        "identity": BASE_DIR / "models2",
        "extracted_frames": BASE_DIR / "extracted_frames"
    }

settings = Settings()

# Create directories
for directory in [settings.IMAGE_UPLOAD_DIR, settings.VIDEO_UPLOAD_DIR, 
                  settings.IMAGE_HEATMAP_DIR, settings.VIDEO_HEATMAP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Create training output directories
for directory in settings.TRAINING_OUTPUT_DIRS.values():
    directory.mkdir(parents=True, exist_ok=True)


# Load config từ file khi khởi động
try:
    from app.routers.training_config_router import load_config_from_file
    load_config_from_file()
    print("✓ Training config loaded from file")
except Exception as e:
    print(f"! Could not load training config: {e}")
# app/routers/training_config_router.py
from pathlib import Path
import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Any
from app.core.config import settings, BASE_DIR

templates = Jinja2Templates(directory="app/templates")

class TrainingConfigRouter:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/training-config", self.config_page, methods=["GET"], response_class=HTMLResponse)
        self.router.add_api_route("/api/training-config", self.get_training_config, methods=["GET"])
        self.router.add_api_route("/api/training-config", self.update_training_config, methods=["PUT"])
        self.router.add_api_route("/api/available-models", self.get_available_models, methods=["GET"])
        
        # Config file path
        self.CONFIG_FILE = BASE_DIR / "app" / "config" / "training_config.json"

    def save_config_to_file(self):
        """Lưu config ra file JSON"""
        try:
            config_data = {
                "IMAGE_TRAINING_CONFIG": settings.IMAGE_TRAINING_CONFIG,
                "VIDEO_TRAINING_CONFIG": settings.VIDEO_TRAINING_CONFIG,
                "IMAGE_MODEL_PATH": str(settings.IMAGE_MODEL_PATH),
                "VIDEO_MODEL_PATH": str(settings.VIDEO_MODEL_PATH),
                "VIDEO_POOLING_STRATEGY": settings.VIDEO_POOLING_STRATEGY
            }
            
            self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Config saved to {self.CONFIG_FILE}")
        except Exception as e:
            print(f"! Error saving config: {e}")

    def load_config_from_file(self):
        """Load config từ file JSON"""
        try:
            if self.CONFIG_FILE.exists():
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Cập nhật settings
                if "IMAGE_TRAINING_CONFIG" in config_data:
                    settings.IMAGE_TRAINING_CONFIG.update(config_data["IMAGE_TRAINING_CONFIG"])
                if "VIDEO_TRAINING_CONFIG" in config_data:
                    settings.VIDEO_TRAINING_CONFIG.update(config_data["VIDEO_TRAINING_CONFIG"])
                if "IMAGE_MODEL_PATH" in config_data:
                    settings.IMAGE_MODEL_PATH = Path(config_data["IMAGE_MODEL_PATH"])
                if "VIDEO_MODEL_PATH" in config_data:
                    settings.VIDEO_MODEL_PATH = Path(config_data["VIDEO_MODEL_PATH"])
                if "VIDEO_POOLING_STRATEGY" in config_data:
                    settings.VIDEO_POOLING_STRATEGY = config_data["VIDEO_POOLING_STRATEGY"]
                    
                print("✓ Training config loaded from file")
        except Exception as e:
            print(f"! Could not load training config: {e}")

    async def config_page(self, request: Request):
        username = request.cookies.get("username")  # lấy từ cookie
        return templates.TemplateResponse("settings.html", {"request": request,"username":username})
    async def get_training_config(self):
        """Lấy toàn bộ config training"""
        return {
            "image": settings.IMAGE_TRAINING_CONFIG,
            "video": settings.VIDEO_TRAINING_CONFIG,
            "image_model_path": str(settings.IMAGE_MODEL_PATH),  # ✅ Thêm model paths
            "video_model_path": str(settings.VIDEO_MODEL_PATH),  # ✅ Thêm model paths
            "video_pooling_strategy": settings.VIDEO_POOLING_STRATEGY  # ✅ Thêm pooling strategy
        }

    async def update_training_config(self, update: dict):
        """Cập nhật config training"""
        try:
            config_type = update.get("config_type")
            updates = update.get("updates", {})
            
            print(f"Updating {config_type} with: {updates}")  # Debug log
            
            if config_type == "image":
                config_dict = settings.IMAGE_TRAINING_CONFIG
                self._apply_config_updates(config_dict, updates)
                
            elif config_type == "video":
                config_dict = settings.VIDEO_TRAINING_CONFIG
                self._apply_config_updates(config_dict, updates)
                
            elif config_type == "model_paths":
                if "image" in updates:
                    settings.IMAGE_MODEL_PATH = Path(updates["image"])
                if "video" in updates:
                    settings.VIDEO_MODEL_PATH = Path(updates["video"])
                if "video_pooling" in updates:
                    settings.VIDEO_POOLING_STRATEGY = updates["video_pooling"]
                    
            else:
                raise HTTPException(status_code=400, detail="Invalid config_type")
            
            # Lưu config ra file
            self.save_config_to_file()
            
            return {"message": f"{config_type} config updated successfully"}
        
        except Exception as e:
            print(f"Error updating config: {e}")  # Debug log
            raise HTTPException(status_code=500, detail=str(e))

    def _apply_config_updates(self, config_dict: dict, updates: dict):
        """Áp dụng các updates vào config dictionary"""
        for key, value in updates.items():
            keys = key.split('.')
            current = config_dict
            
            # Đi đến level cuối cùng
            for k in keys[:-1]:
                if k in current and isinstance(current[k], dict):
                    current = current[k]
                else:
                    print(f"Invalid key path: {key}")  # Debug log
                    raise ValueError(f"Invalid config key: {key}")
            
            # Cập nhật giá trị (chuyển đổi sang int nếu là số)
            final_key = keys[-1]
            if isinstance(value, str) and value.isdigit():
                value = int(value)
            current[final_key] = value

    async def get_available_models(self):
        """Lấy danh sách model có sẵn"""
        models_dir = Path("app/models")
        
        image_models = []
        video_models = []
        
        # Get image models
        image_dir = models_dir / "image"
        if image_dir.exists():
            for model_folder in image_dir.iterdir():
                if model_folder.is_dir():
                    model_file = model_folder / "model_final.h5"
                    if model_file.exists():
                        image_models.append({
                            "name": model_folder.name,
                            "path": str(model_file)
                        })
        
        # Get video models  
        video_dir = models_dir / "video"
        if video_dir.exists():
            for model_folder in video_dir.iterdir():
                if model_folder.is_dir():
                    model_file = model_folder / "model_final.h5"
                    if model_file.exists():
                        video_models.append({
                            "name": model_folder.name,
                            "path": str(model_file)
                        })
        
        return {
            "image": image_models,
            "video": video_models
        }

# Export instance để main.py import
training_config_router = TrainingConfigRouter().router
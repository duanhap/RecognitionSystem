from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from pathlib import Path  # Đảm bảo import từ pathlib
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.training_service import TrainingService
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/templates")

class TrainingRouter:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/train", self.train_page, methods=["GET"], response_class=HTMLResponse)
        self.router.add_api_route("/train/start", self.start_training, methods=["POST"])
        self.router.add_api_route("/train/save", self.save_training, methods=["POST"])
        self.router.add_api_route("/train/cancel", self.cancel_training, methods=["POST"])
        self.router.add_api_route("/train/stop", self.stop_training, methods=["POST"])
        self.router.add_api_route("/train/status", self.training_status, methods=["GET"])
        self.router.add_api_route("/train/available-models", self.get_available_models, methods=["GET"])

    async def train_page(self, request: Request, db: Session = Depends(get_db)):
        # Lấy danh sách model có sẵn
        available_models = await self.get_available_models_data()
        return templates.TemplateResponse(
            "train.html", 
            {
                "request": request,
                "available_models": available_models
            }
        )


    async def training_status(self, request: Request, db: Session = Depends(get_db)):
        """Kiểm tra trạng thái training hiện tại"""
        service = TrainingService(db)
        return {
            "is_running": service._is_running,
            "has_process": service._current_process is not None
        }

    async def save_training(self, request: Request, db: Session = Depends(get_db)):
        service = TrainingService(db)
        
        # LẤY MODEL_PATH TỪ SESSION
        model_path = request.session.get("last_model_path")
        if model_path:
            service._current_result_id = model_path
        
        result = service.save_model()
        
        # XÓA SESSION SAU KHI SAVE
        if "last_model_path" in request.session:
            del request.session["last_model_path"]
        
        return templates.TemplateResponse(
            "train.html",
            {
                "request": request, 
                "message": result.get("message", result.get("error", "Model saved successfully!")),
                "result": result
            }
        )

    async def start_training(
        self, 
        request: Request,
        train_type: str = Form(...),
        num_samples: int = Form(...),
        sample_mode: str = Form(None),
        train_depth: str = Form(...),
        train_mode: str = Form(...),
        resume_model: str = Form(None),
        db: Session = Depends(get_db)
    ):
        userId = request.session.get("user_id")
        service = TrainingService(db)
        
        result = await service.start_training(
            train_type, num_samples, sample_mode, train_depth, 
            train_mode, resume_model, userId
        )

        # LƯU MODEL_PATH VÀO SESSION
        if result.get("model_path"):
            request.session["last_model_path"] = result["model_path"]

        return templates.TemplateResponse(
            "train.html",
            {
                "request": request,
                "result": result,
                "message": result.get("message", "Training completed!"),
                "model_path": result.get("model_path")
            }
        )

    async def cancel_training(self, request: Request):
        return RedirectResponse(url="/train", status_code=303)

    async def stop_training(self, request: Request, db: Session = Depends(get_db)):
        service = TrainingService(db)
        result = service.stop_training()
        return templates.TemplateResponse(
            "train.html",
            {"request": request, "message": result.get("message", "Training stopped!")}
        )
    
    async def get_available_models_data(self):
        """Lấy danh sách model có sẵn từ thư mục models"""
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
                            "path": str(model_file),
                            "type": "image"
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
                            "path": str(model_file),
                            "type": "video"
                        })
        
        return {
            "image": image_models,
            "video": video_models
        }

    async def get_available_models(self, request: Request):
        """API endpoint để lấy danh sách model"""
        models = await self.get_available_models_data()
        return models

training_router = TrainingRouter().router
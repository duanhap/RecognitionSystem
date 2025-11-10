# app/routers/identity_training_router.py
from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.identity_training_service import IdentityTrainingService
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path

templates = Jinja2Templates(directory="app/templates")

class IdentityTrainingRouter:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/train-identity", self.train_page, methods=["GET"], response_class=HTMLResponse)
        self.router.add_api_route("/train-identity/start", self.start_training, methods=["POST"])
        self.router.add_api_route("/train-identity/save", self.save_training, methods=["POST"])
        self.router.add_api_route("/train-identity/discard", self.discard_training, methods=["POST"])

    async def train_page(self, request: Request, db: Session = Depends(get_db)):
        # Lấy danh sách model có sẵn từ thư mục models2
        available_models = self.get_available_models()
        
        username = request.cookies.get("username")
        
        return templates.TemplateResponse(
            "train_identity.html", 
            {
                "request": request,
                "available_models": available_models,
                "username": username
            }
        )

    async def start_training(
        self, 
        request: Request,
        train_mode: str = Form(...),
        resume_model: str = Form(None),
        db: Session = Depends(get_db)
    ):
        username = request.cookies.get("username")
        user_id = request.cookies.get("user_id")
        
        service = IdentityTrainingService(db)
        
        result = await service.start_training(
            train_mode=train_mode,
            resume_model=resume_model,
            user_id=user_id
        )

        # LƯU MODEL_PATH VÀO SESSION SAU KHI TRAINING THÀNH CÔNG
        if result.get("completed") and result.get("model_path"):
            request.session["last_identity_model_path"] = result["model_path"]
            print(f"✅ Saved model path to session: {result['model_path']}")

        available_models = self.get_available_models()
        
        return templates.TemplateResponse(
            "train_identity.html",
            {
                "request": request,
                "result": result,
                "message": result.get("message", ""),
                "available_models": available_models,
                "username": username
            }
        )

    async def save_training(self, request: Request, db: Session = Depends(get_db)):
        username = request.cookies.get("username")
        service = IdentityTrainingService(db)
        
        # Lấy model_path từ session
        model_path = request.session.get("last_identity_model_path")
        print(f"🔍 Retrieving model path from session: {model_path}")
        
        if not model_path:
            available_models = self.get_available_models()
            return templates.TemplateResponse(
                "train_identity.html",
                {
                    "request": request,
                    "message": "No training result to save! Please run training first.",
                    "available_models": available_models,
                    "username": username
                }
            )
        
        # Set model path cho service
        service._current_model_path = model_path
        
        result = service.save_training_results()
        
        # Xóa session sau khi save thành công
        if "last_identity_model_path" in request.session:
            del request.session["last_identity_model_path"]
            print("🗑️ Removed model path from session after save")
        
        available_models = self.get_available_models()
        
        return templates.TemplateResponse(
            "train_identity.html",
            {
                "request": request,
                "message": result.get("message", result.get("error", "Training results saved successfully!")),
                "available_models": available_models,
                "username": username
            }
        )

    async def discard_training(self, request: Request, db: Session = Depends(get_db)):
        username = request.cookies.get("username")
        service = IdentityTrainingService(db)
        
        # Reset service state
        service.discard_training()
        
        # Xóa session
        if "last_identity_model_path" in request.session:
            del request.session["last_identity_model_path"]
            print("🗑️ Removed model path from session after discard")
        
        available_models = self.get_available_models()
        
        return templates.TemplateResponse(
            "train_identity.html",
            {
                "request": request,
                "message": "Training discarded successfully!",
                "available_models": available_models,
                "username": username
            }
        )

    def get_available_models(self):
        """Lấy danh sách model có sẵn từ thư mục models2"""
        app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        models_dir = Path(os.path.join(app_dir, "models2"))
        
        available_models = []
        
        if models_dir.exists():
            for model_folder in models_dir.iterdir():
                if model_folder.is_dir():
                    model_file = model_folder / "model_final.h5"
                    if model_file.exists():
                        available_models.append({
                            "name": model_folder.name,
                            "path": str(model_file)
                        })
        
        # Sắp xếp theo thời gian (mới nhất trước)
        available_models.sort(key=lambda x: x["name"], reverse=True)
        return available_models

identity_training_router = IdentityTrainingRouter().router
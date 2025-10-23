from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse
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


    async def train_page(self, request: Request, db: Session = Depends(get_db)):
        # Hiển thị giao diện train.html
        return templates.TemplateResponse("train.html", {"request": request})

    async def start_training(
        self, 
        request: Request,
        train_type: str = Form(...),
        num_samples: int = Form(...),
        train_depth: str = Form(...),
        train_mode: str = Form(...),
        db: Session = Depends(get_db)
    ):
        service = TrainingService(db)
        result = service.start_training(train_type, num_samples, train_depth, train_mode)

        # render lại trang train.html, show kết quả
        return templates.TemplateResponse(
            "train.html",
            {
                "request": request,
                "result": result,
                "message": "Training completed!"
            }
        )
    async def save_training(self, request: Request, db: Session = Depends(get_db)):
        service = TrainingService(db)
        service.save_model()   # lưu file h5/pth + DB record
        return templates.TemplateResponse(
            "train.html",
            {"request": request, "message": "Model đã được lưu thành công!"}
        )

    async def cancel_training(self, request: Request):
        # Không lưu gì, quay về trang train
        return RedirectResponse(url="/train", status_code=303)

    async def stop_training(self, request: Request, db: Session = Depends(get_db)):
        service = TrainingService(db)
        service.stop_training()  # dừng vòng lặp huấn luyện
        return templates.TemplateResponse(
            "train.html",
            {"request": request, "message": "Training đã dừng lại!"}
        )

training_router = TrainingRouter().router

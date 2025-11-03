from typing import List
from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.params import Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from fastapi import File
from app.services.sample_service import SampleService
from app.core.database import get_db

templates = Jinja2Templates(directory="app/templates")

class AddMultipleSamplesRouter:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/add_multiple_samples", self.add_sample_page, methods=["GET"], response_class=HTMLResponse)
        self.router.add_api_route("/add_multiple_samples", self.add_sample, methods=["POST"])

    async def add_sample_page(self, request: Request):
        username = request.cookies.get("username")  # lấy từ cookie
        return templates.TemplateResponse("add_multiple_samples.html", {"request": request, "username": username})

    async def add_sample(
        self, 
        request: Request,
        type: str = Form(...), 
        label: str = Form(...), 
        description: str = Form(...), 
        files: List[UploadFile] = File(...), 
        db: Session = Depends(get_db)
    ):
        user_id = request.cookies.get("user_id")
        if not user_id:
            # nếu chưa login thì chuyển về trang login
            return RedirectResponse(url="/login", status_code=303)

        service = SampleService(db)
        success = service.create_samples(type, label, description, files, user_id)

        # Ở lại trang add_multiple_samples.html, thêm message
        return templates.TemplateResponse(
            "add_multiple_samples.html",
            {
                "request": request,
                "message": "Samples added successfully!" if success else "Failed to add sample."
            }
    )

add_multiple_samples_router = AddMultipleSamplesRouter().router

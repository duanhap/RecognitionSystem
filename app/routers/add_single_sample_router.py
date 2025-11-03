import json
from typing import List
from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.params import Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from fastapi import File
from app.services.sample_service import SampleService
from app.core.database import get_db

templates = Jinja2Templates(directory="app/templates")

class AddSingleSampleRouter:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/add_single_sample", self.add_single_sample_page, methods=["GET"], response_class=HTMLResponse)
        self.router.add_api_route("/add_single_sample", self.add_single_sample, methods=["POST"])

    async def add_single_sample_page(self, request: Request):
        username = request.cookies.get("username")  # lấy từ cookie
        return templates.TemplateResponse("add_single_sample.html", {"request": request, "username": username})

    async def add_single_sample(
        self, 
        request: Request,
        type: str = Form(...), 
        label: str = Form(...), 
        description: str = Form(...),
        crop_data : str = Form(None), 
        file: UploadFile = File(...), 
        db: Session = Depends(get_db)
    ):
        print(f"Received file: {file.filename if file else 'None'}")  # Log để check
        print(f"Crop data: {crop_data}")
        
        if not file:
            return JSONResponse(status_code=400, content={"message": "No file uploaded"})
        
        user_id = request.cookies.get("user_id")
        if not user_id:
            return RedirectResponse(url="/login", status_code=303)
        
        try:
            crop_info = json.loads(crop_data) if crop_data else None
            service = SampleService(db)
            success = service.create_sample(type, label, description, file, user_id, crop_info)
            if success:
                return templates.TemplateResponse(
                    "add_single_sample.html",
                    {"request": request, "message": "Sample added successfully!"}
                )
            else:
                raise ValueError("Failed to add sample")
        except Exception as e:
            print(f"Error: {str(e)}")  # Log error
            return templates.TemplateResponse(
                "add_single_sample.html",
                {"request": request, "message": f"Error: {str(e)}"}
            )
add_single_sample_router = AddSingleSampleRouter().router
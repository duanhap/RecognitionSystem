from fastapi import APIRouter, Depends, Request , Form, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from app.services.sample_service import SampleService
from app.core.database import get_db

templates = Jinja2Templates(directory="app/templates")

class ViewDetailSampleRouter:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/view_sample/{sample_id}", self.view_sample_page, methods=["GET"], response_class=HTMLResponse)
        self.router.add_api_route("/view_sample/{sample_id}/update", self.update_sample, methods=["POST"])
        self.router.add_api_route("/view_sample/{sample_id}/delete", self.delete_sample, methods=["GET"])

    async def view_sample_page(self, request: Request, sample_id: int, db: Session = Depends(get_db)):
        service = SampleService(db)
        sample = service.get_sample(sample_id)
        return templates.TemplateResponse("view_detail_sample.html", {"request": request, "sample": sample})

    async def update_sample(
        self, 
        request: Request, 
        sample_id: int, 
        label: str = Form(...), 
        description: str = Form(...), 
        db: Session = Depends(get_db)
    ):
        service = SampleService(db)
        
        # Lấy sample hiện tại để lấy type
        sample = service.get_sample(sample_id)
        if not sample:
            return templates.TemplateResponse("error.html", {"request": request, "message": "Sample not found"})
        
        success = service.update_sample(sample_id, sample.type, label, description)
        if success:
            return RedirectResponse(url=f"/view_sample/{sample_id}", status_code=303)
        else:
            return templates.TemplateResponse("error.html", {"request": request, "message": "Failed to update sample"})


    async def delete_sample(self, request: Request, sample_id: int, db: Session = Depends(get_db)):
        service = SampleService(db)
        success = service.remove_sample(sample_id)
        if success:
            return RedirectResponse(url="/manage_samples", status_code=303)
        else:
            return templates.TemplateResponse("error.html", {"request": request, "message": "Sample not found"})

view_detail_sample_router = ViewDetailSampleRouter().router

from math import ceil
from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from app.services.sample_service import SampleService
from app.core.database import get_db


templates = Jinja2Templates(directory="app/templates")

class ManageSampleRouter:
    def __init__(self):
        self.router = APIRouter()
        # Trang quản lý danh sách
        self.router.add_api_route("/manage_samples", self.manage_samples_page, methods=["GET"], response_class=HTMLResponse)
        # Xóa sample
        self.router.add_api_route("/manage_samples/delete/{sample_id}", self.delete_sample, methods=["GET"])

    async def manage_samples_page(self, request: Request, search: str = None, page:int=1, db: Session = Depends(get_db)):
        service = SampleService(db)
        per_page = 10
        samples, total = service.search_samples(search, page=page, per_page=per_page)
        total_pages = ceil(total / per_page)

        return templates.TemplateResponse("manage_samples.html", {
            "request": request,
            "samples": samples,
            "search": search,
            "page": page,
            "total_pages": total_pages
        })

    async def delete_sample(self, request: Request, sample_id: int, db: Session = Depends(get_db)):
        service = SampleService(db)
        success = service.remove_sample(sample_id)
        if success:
            return RedirectResponse(url="/manage_samples", status_code=303)
        else:
            return templates.TemplateResponse("error.html", {"request": request, "message": "Sample not found"})

manage_sample_router = ManageSampleRouter().router

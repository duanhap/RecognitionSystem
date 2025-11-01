# app/routers/view_detail_result_router.py
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.services.analysis_result_service import AnalysisResultService
from app.core.database import get_db

templates = Jinja2Templates(directory="app/templates")

class ViewDetailResultRouter:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/view-detail-result/{result_id}", self.view_result_page, methods=["GET"], response_class=HTMLResponse)

    async def view_result_page(self, request: Request, result_id: int, db: Session = Depends(get_db)):
        service = AnalysisResultService(db)
        
        # Lấy thông tin chi tiết kết quả
        result = service.get_analysis_result(result_id)
        
        if not result:
            # Trả về trang lỗi nếu không tìm thấy
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error_message": "Result not found"
            })
        
        return templates.TemplateResponse("view_detail_test_result.html", {
            "request": request,
            "result": result
        })

# Export instance
view_detail_result_router = ViewDetailResultRouter().router
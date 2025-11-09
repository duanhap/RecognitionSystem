# app/routers/manage_test_result_router.py
from fastapi import APIRouter, Depends, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.services.analysis_result_service import AnalysisResultService
from app.core.database import get_db

templates = Jinja2Templates(directory="app/templates")

class ManageTestResultRouter:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/manage-test-results", self.manage_result_page, methods=["GET"], response_class=HTMLResponse)

    async def manage_result_page(
        self, 
        request: Request, 
        search: str = Query("", alias="search"),
        db: Session = Depends(get_db)
    ):
        service = AnalysisResultService(db)
        
        # Lấy danh sách kết quả
        results = service.search_analysis_results(search)
        
        # Tính điểm trung bình
        avg_confidence = service.average_confidence(results)
        username = request.cookies.get("username")  # lấy từ cookie

        
        # Thống kê tổng số uploads
        total_uploads = len(results)
        
        return templates.TemplateResponse("manage_test_results.html", {
            "request": request,
            "results": results,
            "search": search,
            "average_confidence": avg_confidence,
            "total_uploads": total_uploads,
            "username" : username
        })

# Export instance
manage_test_result_router = ManageTestResultRouter().router
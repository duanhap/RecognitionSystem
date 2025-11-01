# app/routers/feedback_router.py
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.services.feedback_service import FeedbackService
from app.core.database import get_db

templates = Jinja2Templates(directory="app/templates")

class ViewFeedbackRouter:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/view_feedback", self.view_feedback_page, methods=["GET"], response_class=HTMLResponse)

    async def view_feedback_page(self, request: Request, db: Session = Depends(get_db)):
        service = FeedbackService(db)
        
        # Lấy danh sách feedback
        feedbacks = service.get_feedbacks()
        
        # Tính điểm trung bình
        average_point = service.get_average_point(feedbacks)
        
        # Tính phân phối rating
        rating_distribution = service.get_rating_distribution(feedbacks)
        
        return templates.TemplateResponse("view_feedback.html", {
            "request": request,
            "feedbacks": feedbacks,
            "average_rating": average_point,
            "total_feedbacks": len(feedbacks),
            "rating_distribution": rating_distribution
        })

# Export instance
feedback_router = ViewFeedbackRouter().router
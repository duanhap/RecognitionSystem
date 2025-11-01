# app/services/feedback_service.py
from typing import List, Dict
from app.repositories.feedback_repository import FeedbackRepository
from app.model.feedback import Feedback

class FeedbackService:
    def __init__(self, db):
        self.feedback_repo = FeedbackRepository(db)

    def get_feedbacks(self) -> List[Feedback]:
        """Lấy toàn bộ danh sách feedback"""
        return self.feedback_repo.list_feedbacks()

    def get_average_point(self, feedbacks: List[Feedback]) -> float:
        """Tính điểm đánh giá trung bình"""
        if not feedbacks:
            return 0.0
        
        total_rating = sum(feedback.rating for feedback in feedbacks)
        return round(total_rating / len(feedbacks), 1)

    def get_rating_distribution(self, feedbacks: List[Feedback]) -> Dict[int, float]:
        """Tính phân phối rating theo sao"""
        distribution = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
        
        if not feedbacks:
            return distribution
        
        for feedback in feedbacks:
            rating = int(feedback.rating)
            if rating in distribution:
                distribution[rating] += 1
        
        # Tính phần trăm
        total = len(feedbacks)
        return {stars: round((count / total) * 100, 1) for stars, count in distribution.items()}
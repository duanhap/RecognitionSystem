# app/repositories/feedback_repository.py
from typing import List
from sqlalchemy.orm import Session
from app.model.feedback import Feedback
from app.model.user import User

class FeedbackRepository:
    def __init__(self, db: Session):
        self.db = db

    def list_feedbacks(self) -> List[Feedback]:
        return self.db.query(Feedback)\
            .join(User, Feedback.user_id == User.id)\
            .order_by(Feedback.created_at.desc())\
            .all()
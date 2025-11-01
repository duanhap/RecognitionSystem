# app/models/feedback.py (cập nhật)
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.model.analysis_result import AnalysisResult
from datetime import datetime
from app.core.database import database

class Feedback(database.Base):
    __tablename__ = "tblFeedback"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String(255))
    rating = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Quan hệ với User
    user_id = Column(Integer, ForeignKey("tblUser.id"), nullable=False)
    user = relationship("User", back_populates="feedbacks")

    # Quan hệ với AnalysisResult
    analysis_result_id = Column(Integer, ForeignKey("tblAnalysisResult.id"), nullable=False)
    analysis_result = relationship("AnalysisResult", back_populates="feedback", uselist=False)

    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "rating": self.rating,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "user_id": self.user_id,
            "analysis_result_id": self.analysis_result_id,
            "username": self.user.username if self.user else None
        }
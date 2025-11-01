from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.model.media import Media
from datetime import datetime
from app.core.database import database

class AnalysisResult(database.Base):
    __tablename__ = "tblAnalysisResult"

    id = Column(Integer, primary_key=True, index=True)
    confidence_score = Column(Float)
    heatmap_path = Column(String(255))
    label = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Quan hệ với Media
    media_id = Column(Integer, ForeignKey("tblMedia.id"), nullable=False)
    media = relationship("Media", back_populates="analysis_result", uselist=False)

    # Quan hệ với Feedback
    feedback= relationship("Feedback", back_populates="analysis_result",uselist=False)

    def to_dict(self):
        return {
            "id": self.id,
            "confidence_score": self.confidence_score,
            "heatmap_path": self.heatmap_path,
            "label": self.label,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "media_id": self.media_id
        }

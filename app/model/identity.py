# app/models/identity.py
from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from app.core.database import database
from sqlalchemy.orm import relationship

class Identity(database.Base):
    __tablename__ = "tblIdentity"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)          # Tên người (VD: "Nguyen Van A")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Quan hệ 1-n với TrainingSample và AnalysisResult
    training_samples = relationship("TrainingSample", back_populates="identity")
    analysis_results = relationship("AnalysisResult", back_populates="identity")

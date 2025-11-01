from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import database

class Media(database.Base):
    __tablename__ = "tblMedia"

    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String(255), nullable=False)     # đường dẫn file upload
    type = Column(String(50), nullable=True)            # ví dụ: "image", "video"
    upload_at = Column(DateTime, default=datetime.utcnow)
    
    # Quan hệ với User
    user_id = Column(Integer, ForeignKey("tblUser.id"), nullable=False)
    user = relationship("User", backref="tblMedia")

    # Quan hệ với AnalysisResult
    analysis_result = relationship("AnalysisResult", back_populates="media", uselist=False)

    def to_dict(self):
        return {
            "id": self.id,
            "file_path": self.file_path,
            "type": self.type,
            "upload_at": self.upload_at.isoformat() if self.upload_at else None,
            "user_id": self.user_id
        }

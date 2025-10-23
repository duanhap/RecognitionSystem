from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import database

class TrainingSample(database.Base):
    __tablename__ = "training_samples"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String(20), nullable=False)         # "image" hoặc "video"
    label = Column(String(20), nullable=False)        # "real" hoặc "fake"
    description = Column(String(255), nullable=True)  # mô tả: faketype, source
    file_path = Column(String(255), nullable=False)   # đường dẫn file upload
    created_at = Column(DateTime, default=datetime.utcnow)

    # Quan hệ với User
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", backref="training_samples")

# app/models/user.py
from lib2to3.pytree import Base
from sqlalchemy import Column, Integer, String, DateTime , ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import database
from datetime import datetime


class User(database.Base):
    __tablename__ = "tblUser"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=True)
    phone = Column(String(20), nullable=True)
    role = Column(String(20), default="user")        # user / admin
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="active")    # active / inactive
    # Thêm relationship với Feedback
    feedbacks = relationship("Feedback", back_populates="user")

    def to_dict(self):
            return {
                "id": self.id,
                "username": self.username,
                "email": self.email,
                "phone": self.phone,
                "role": self.role,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "status": self.status
            }
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import database
from app.model.identity import Identity

class TrainingSample(database.Base):
    __tablename__ = "tblTrainingSample"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String(20), nullable=False)         
    label = Column(String(20), nullable=False)        
    description = Column(String(255), nullable=True)  
    file_path = Column(String(255), nullable=False)   
    created_at = Column(DateTime, default=datetime.utcnow)

    # Quan hệ với User
    user_id = Column(Integer, ForeignKey("tblUser.id"))
    user = relationship("User", backref="tblTrainingSample")

    # Quan hệ mới: với Identity
    identity_id = Column(Integer, ForeignKey("tblIdentity.id"), nullable=True)
    identity = relationship("Identity", back_populates="training_samples")

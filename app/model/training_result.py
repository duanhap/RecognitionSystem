from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import database

class TrainingResult(database.Base):
    __tablename__ = "tblTrainingResult"

    id = Column(Integer, primary_key=True, index=True)
    acc = Column(Float)
    pre = Column(Float)
    rec = Column(Float)
    f1 = Column(Float)
    file_path = Column(String(255))  # đường dẫn file model đã lưu
    training_sample_id = Column(Integer, ForeignKey("tblTrainingSample.id"))

    sample = relationship("TrainingSample", back_populates="results")
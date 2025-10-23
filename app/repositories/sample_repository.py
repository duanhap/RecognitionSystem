from fastapi import APIRouter, Depends, Request, UploadFile
from sqlalchemy.orm import Session
from app.models.training_sample import TrainingSample
class SampleRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_sample_by_id(self, sample_id: int):
        return self.db.query(TrainingSample).filter(TrainingSample.id == sample_id).first()

    def list_samples(self, search: str = None):
        query = self.db.query(TrainingSample)
        if search:
            query = query.filter(TrainingSample.id == search)  # tìm theo ID
        return query.all()

    def add_sample(self, type: str, label: str, description: str, file: UploadFile, user_id: int):
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        sample = TrainingSample(type=type, label=label, description=description, file_path=file_path, user_id=user_id)
        self.db.add(sample)
        self.db.commit()
        self.db.refresh(sample)
        return sample

    def update_sample(self, sample_id: int, type: str, label: str, description: str) -> bool:
        sample = self.get_sample_by_id(sample_id)
        if not sample:
            return False
        sample.type = type
        sample.label = label
        sample.description = description
        self.db.commit()
        self.db.refresh(sample)
        return True

    def delete_sample(self, sample_id: int) -> bool:
        sample = self.get_sample_by_id(sample_id)
        if not sample:
            return False
        self.db.delete(sample)
        self.db.commit()
        return True

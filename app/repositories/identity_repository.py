# app/repositories/identity_repository.py
from sqlalchemy.orm import Session
from app.model.training_sample import TrainingSample
from app.model.identity import Identity

class IdentityRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_identity_samples(self, search: str = None, skip: int = 0, limit: int = 10):
        query = self.db.query(TrainingSample).filter(
            TrainingSample.file_path.like('dataset2/%')
        )
        
        if search:
            query = query.filter(
                TrainingSample.label.ilike(f'%{search}%') |
                TrainingSample.description.ilike(f'%{search}%')
            )
        
        return query.offset(skip).limit(limit).all()

    def count_identity_samples(self, search: str = None):
        query = self.db.query(TrainingSample).filter(
            TrainingSample.file_path.like('dataset2/%')
        )
        
        if search:
            query = query.filter(
                TrainingSample.label.ilike(f'%{search}%') |
                TrainingSample.description.ilike(f'%{search}%')
            )
        
        return query.count()

    def get_all_identities(self):
        """Lấy tất cả identities"""
        return self.db.query(Identity).all()
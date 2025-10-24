from typing import List
from fastapi import UploadFile
from app.repositories.sample_repository import SampleRepository
class SampleService:
    def __init__(self, db):
        self.db = db
        self.sample_repo = SampleRepository(db)

    def get_sample(self, sample_id: int):
        return self.sample_repo.get_sample_by_id(sample_id)

    def search_samples(self, search: str = None):
        return self.sample_repo.list_samples(search)

    def create_sample(self, type: str, label: str, description: str, file: UploadFile, user_id: int, crop_info: dict = None) -> bool:
        try:
            self.sample_repo.add_sample(type, label, description, file, user_id, crop_info)
            return True
        except Exception as e:
            print(f"❌ Error adding samples: {e}")
            return False 
    def create_samples(self, type: str, label: str, description: str, files: List[UploadFile], user_id: int):
        try:
            for file in files:
                self.sample_repo.add_sample(type, label, description, file, user_id)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            print(f"❌ Error adding samples: {e}")
            return False

    def update_sample(self, sample_id: int, type: str, label: str, description: str) -> bool:
        return self.sample_repo.update_sample(sample_id, type, label, description)

    def remove_sample(self, sample_id: int) -> bool:
        return self.sample_repo.delete_sample(sample_id)

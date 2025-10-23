from fastapi import UploadFile
from app.repositories.sample_repository import SampleRepository
class SampleService:
    def __init__(self, db):
        self.sample_repo = SampleRepository(db)

    def get_sample(self, sample_id: int):
        return self.sample_repo.get_sample_by_id(sample_id)

    def search_samples(self, search: str = None):
        return self.sample_repo.list_samples(search)

    def create_sample(self, type: str, label: str, description: str, file: UploadFile, user_id: int):
        return self.sample_repo.add_sample(type, label, description, file, user_id=user_id)

    def update_sample(self, sample_id: int, type: str, label: str, description: str) -> bool:
        return self.sample_repo.update_sample(sample_id, type, label, description)

    def remove_sample(self, sample_id: int) -> bool:
        return self.sample_repo.delete_sample(sample_id)

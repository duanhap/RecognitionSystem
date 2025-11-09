# app/services/identity_service.py
import os
import shutil
from datetime import datetime
from typing import List, Tuple
from fastapi import UploadFile
from sqlalchemy.orm import Session
from app.model.training_sample import TrainingSample
from app.model.identity import Identity
from app.repositories.identity_repository import IdentityRepository

class IdentityService:
    def __init__(self, db: Session):
        self.db = db
        self.identity_repo = IdentityRepository(db)

    def get_identity_samples(self, search: str = None, page: int = 1, per_page: int = 10) -> Tuple[List[TrainingSample], int, int]:
        """Lấy samples từ dataset2 (identity verification)"""
        skip = (page - 1) * per_page
        samples = self.identity_repo.get_identity_samples(search, skip, per_page)
        total = self.identity_repo.count_identity_samples(search)
        total_pages = (total + per_page - 1) // per_page
        return samples, total, total_pages

    def get_all_identities(self):
        """Lấy tất cả identities"""
        return self.identity_repo.get_all_identities()

    def create_identity(self, name: str) -> bool:
        """Tạo identity mới"""
        try:
            identity = Identity(name=name)
            self.db.add(identity)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            print(f"Error creating identity: {e}")
            return False

    def add_identity_sample(self, identity_id: int, file_type: str, description: str, file: UploadFile, user_id: int) -> bool:
        """Thêm sample cho identity verification - hỗ trợ nhiều file"""
        try:
            # Lấy identity name để dùng làm label
            identity = self.db.query(Identity).filter(Identity.id == identity_id).first()
            if not identity:
                return False

            # Tạo đường dẫn cho dataset2
            base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset2")
            save_dir = os.path.join(base_dir, file_type, identity.name)
            os.makedirs(save_dir, exist_ok=True)

            # Tạo tên file với timestamp để tránh trùng lặp
            timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            filename = f"{timestamp}_{file.filename}"
            file_path = os.path.join(save_dir, filename)
            
            # Lưu file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Tạo file_path với prefix dataset2/
            rel_path = f"dataset2/{file_type}/{identity.name}/{filename}"

            # Lưu vào database
            sample = TrainingSample(
                type=file_type,
                label=identity.name,
                description=description,
                file_path=rel_path,
                user_id=user_id,
                identity_id=identity_id
            )
            self.db.add(sample)
            self.db.commit()
            return True

        except Exception as e:
            self.db.rollback()
            print(f"Error adding identity sample: {e}")
            return False

    def delete_sample(self, sample_id: int) -> bool:
        """Xóa sample và file vật lý"""
        sample = self.db.query(TrainingSample).filter(TrainingSample.id == sample_id).first()
        if not sample:
            return False

        try:
            # Xóa file vật lý
            base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)))
            file_path = os.path.join(base_dir, sample.file_path)
            if os.path.exists(file_path):
                os.remove(file_path)

            # Xóa trong database
            self.db.delete(sample)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            print(f"Error deleting sample: {e}")
            return False
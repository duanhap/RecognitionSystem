import os
import shutil
from typing import List
from fastapi import UploadFile
from app.repositories.sample_repository import SampleRepository
class SampleService:
    def __init__(self, db):
        self.db = db
        self.sample_repo = SampleRepository(db)
    def update_sample(self, sample_id: int, type: str, label: str, description: str) -> bool:
        # Lấy sample hiện tại
        sample = self.sample_repo.get_sample_by_id(sample_id)
        if not sample:
            return False
            
        old_label = sample.label
        old_file_path = sample.file_path
        
        # Cập nhật trong CSDL
        success = self.sample_repo.update_sample(sample_id, type, label, description)
        if not success:
            return False
            
        # Nếu label thay đổi, di chuyển file vật lý
        if old_label != label:
            self._move_sample_file(sample_id, old_file_path, old_label, label, type)
            
        return True

    def _move_sample_file(self, sample_id: int, old_file_path: str, old_label: str, new_label: str, file_type: str):
        """Di chuyển file sample từ thư mục cũ sang thư mục mới khi label thay đổi"""
        try:
            # Lấy đường dẫn base dataset
            base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
            
            # Đường dẫn file cũ (tuyệt đối)
            old_abs_path = os.path.join(base_dir, old_file_path)
            
            if not os.path.exists(old_abs_path):
                print(f"File không tồn tại: {old_abs_path}")
                return
                
            # Tạo đường dẫn mới
            filename = os.path.basename(old_file_path)
            new_rel_path = os.path.join(file_type, new_label, filename)
            new_abs_path = os.path.join(base_dir, new_rel_path)
            
            # Tạo thư mục đích nếu chưa tồn tại
            os.makedirs(os.path.dirname(new_abs_path), exist_ok=True)
            
            # Di chuyển file
            shutil.move(old_abs_path, new_abs_path)
            print(f"✅ Đã di chuyển file: {old_abs_path} -> {new_abs_path}")
            
            # Cập nhật file_path trong CSDL
            self.sample_repo.update_file_path(sample_id, new_rel_path)
            
        except Exception as e:
            print(f"Lỗi khi di chuyển file sample {sample_id}: {e}")

    def get_sample(self, sample_id: int):
        return self.sample_repo.get_sample_by_id(sample_id)

    def search_samples(self, search: str = None, page: int = 1, per_page: int = 10):
        skip = (page - 1) * per_page
        samples = self.sample_repo.list_samples(search, skip=skip, limit=per_page)
        total = self.sample_repo.count_samples(search)
        return samples, total

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



    def remove_sample(self, sample_id: int) -> bool:
        return self.sample_repo.delete_sample(sample_id)

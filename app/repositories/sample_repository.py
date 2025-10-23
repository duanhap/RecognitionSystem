from datetime import datetime
import os
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
        # --- 1. Tạo đường dẫn thư mục dựa theo loại file và nhãn ---
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "dataset")
        # => từ repositories/ lên app/, rồi lên project/, rồi tới dataset/
        save_dir = os.path.join(base_dir, type, label)

        # --- 2. Tạo thư mục nếu chưa có ---
        os.makedirs(save_dir, exist_ok=True)

        # --- 3. Đặt tên file an toàn (thêm timestamp để tránh trùng) ---
        filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        file_path = os.path.join(save_dir, filename)

        # --- 4. Ghi file ---
        with open(file_path, "wb") as f:
            while chunk := file.file.read(1024 * 1024):
                f.write(chunk)

        # --- 5. Lưu thông tin vào DB ---
        sample = TrainingSample(
            type=type,
            label=label,
            description=description,
            file_path=file_path,  # lưu đường dẫn tuyệt đối hoặc tương đối đều được
            user_id=user_id
        )
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

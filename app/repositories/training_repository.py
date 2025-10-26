from datetime import datetime  
from app.model.training_result import TrainingResult
from app.model.training_sample import TrainingSample
from sqlalchemy.orm import Session

class TrainingRepository:
    def __init__(self, db: Session):
        self.db = db

    # Lấy samples để train
    def get_samples(self, train_type: str, num_samples: int):
        query = self.db.query(TrainingSample).filter(TrainingSample.type == train_type)
        return query.limit(num_samples).all()

    def save_training_result(self, accuracy: float, precision: float, recall: float, f1: float, sample_id: int, user_id: int = None, model_path: str = None):
        result = TrainingResult(
            acc=accuracy,
            pre=precision,
            rec=recall,
            f1=f1,
            file_path=model_path,  # THÊM DÒNG NÀY
            created_at=datetime.utcnow(),
            training_sample_id=sample_id
        )
        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)
        return result

    # Lưu đường dẫn model sau khi save_model()
    def save_model_file(self, result_id: int, model_path: str):
        result = self.db.query(TrainingResult).filter(TrainingResult.id == result_id).first()
        if not result:
            return None
        result.model_path = model_path
        self.db.commit()
        self.db.refresh(result)
        return result

    # Xoá kết quả training (cancel)
    def delete_training_result(self, result_id: int):
        result = self.db.query(TrainingResult).filter(TrainingResult.id == result_id).first()
        if not result:
            return False
        self.db.delete(result)
        self.db.commit()
        return True

    # Lấy chi tiết kết quả training
    def get_result_by_id(self, result_id: int):
        return self.db.query(TrainingResult).filter(TrainingResult.id == result_id).first()

    # List tất cả kết quả training (cho admin / dashboard)
    def list_results(self, user_id: int = None):
        query = self.db.query(TrainingResult)
        if user_id:
            query = query.filter(TrainingResult.user_id == user_id)
        return query.order_by(TrainingResult.created_at.desc()).all()

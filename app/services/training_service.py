from app.repositories.training_repository import TrainingRepository
import random

class TrainingService:
    def __init__(self, db):
        self.repo = TrainingRepository(db)
        self._current_result_id = None  # lưu tạm id training để save/cancel

    def start_training(self, train_type: str, num_samples: int, train_depth: str, train_mode: str, user_id: int = None):
        # 1. Lấy samples từ DB
        samples = self.repo.get_samples(train_type, num_samples)

        if not samples:
            return {"error": "No samples found!"}

        # 2. Giả lập training logic
        if train_depth == "normal":
            acc, pre, rec, f1 = 0.85, 0.80, 0.82, 0.81
        elif train_depth == "deep":
            acc, pre, rec, f1 = 0.90, 0.87, 0.88, 0.89
        else:  # super_deep
            acc, pre, rec, f1 = 0.93, 0.91, 0.92, 0.92

        # 3. Lưu kết quả vào DB (chưa lưu file model)
        result = self.repo.save_training_result(
            accuracy=acc, precision=pre, recall=rec, f1=f1, 
            sample_id=samples[0].id,
            user_id=user_id
        )

        self._current_result_id = result.id

        return {
            "accuracy": acc,
            "precision": pre,
            "recall": rec,
            "f1": f1,
            "result_id": result.id
        }

    def save_model(self):
        if not self._current_result_id:
            return {"error": "No training result to save!"}

        # Gọi repo lưu file model giả lập
        self.repo.save_model_file(self._current_result_id, "models/model_latest.pth")

        return {"message": "Model saved successfully!"}

    def cancel_training(self):
        if self._current_result_id:
            self.repo.delete_training_result(self._current_result_id)
            self._current_result_id = None
        return {"message": "Training canceled."}

    def stop_training(self):
        # Ở thực tế: stop training loop (PyTorch/TensorFlow)
        return {"message": "Training stopped."}

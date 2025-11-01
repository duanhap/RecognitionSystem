import os
import subprocess
import sys
import asyncio
from app.repositories.training_repository import TrainingRepository
import random
import json
import pandas as pd
from app.repositories.sample_repository import SampleRepository

class TrainingService:
    def __init__(self, db):
        self.db = db
        self.repo = TrainingRepository(db)
        self._current_result_id = None
        self._current_process = None
        self._is_running = False

    async def start_training(self, train_type: str, num_samples: int, sample_mode: str, train_depth: str, train_mode: str, resume_model: str = None, user_id: int = None):
        print(f"Starting training: type={train_type}, samples={num_samples}, depth={train_depth}, mode={train_mode}, user_id={user_id}")
        
        try:
            # Xác định file script theo loại
            script_name = "train_video.py" if train_type == "video" else "train_image.py"
            app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            script_path = os.path.join(app_dir, script_name)

            if not os.path.exists(script_path):
                return {"error": f"Training script not found: {script_path}"}

            # Xây dựng command
            cmd = [
                sys.executable,
                script_path,
                "--dataset_root", f"dataset/{train_type}",
                "--n_samples", str(num_samples),
                "--sampling_mode", sample_mode,
                "--depth", train_depth,
                "--out_root", f"models/{train_type}",
                "--train_ratio", "0.8",
                "--pooling_strategy", "mean",
            ]

            print("🚀 Running command:", " ".join(cmd))

            # Chạy subprocess
            self._current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=app_dir
            )

            self._is_running = True

            # Đọc output
            #stdout, stderr = self._current_process.communicate()

            # Đọc stdout real-time
            for line in iter(self._current_process.stdout.readline, ''):
                line = line.strip()
                if line:
                    print("STDOUT:", line)
                    if "Results saved to:" in line:
                        result_path = line.split("Results saved to:")[-1].strip()
                    elif "Training finished! Results in:" in line:
                        result_path = line.split("Training finished! Results in:")[-1].strip()

            # Đọc stderr real-time
            for line in iter(self._current_process.stderr.readline, ''):
                line = line.strip()
                if line:
                    print("STDERR:", line)
            
            # Đợi process kết thúc
            return_code = self._current_process.wait()
            self._current_process = None
            self._is_running = False
        
            # CHUYỂN ĐƯỜNG DẪN THÀNH ABSOLUTE VÀ CHUẨN HÓA
            if not os.path.isabs(result_path):
                # Nếu là relative path, chuyển thành absolute từ thư mục app
                app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                result_path = os.path.join(app_dir, result_path)

            # CHUẨN HÓA ĐƯỜNG DẪN (thay \ thành /)
            result_path = os.path.normpath(result_path)

            print(f"🔍 Checking model path: {result_path}")  # DEBUG
            
            # Kiểm tra đường dẫn tồn tại
            if not os.path.exists(result_path):
                return {"error": f"Model path not found: {result_path}. Current working dir: {os.getcwd()}"}

            metrics_file = os.path.join(result_path, "metrics.json")

            print(f"🔍 Checking metrics file: {metrics_file}")  # DEBUG
            
            if not os.path.exists(metrics_file):
                return {"error": f"Metrics file not found: {metrics_file}"}
            

            # Đọc metrics.json
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        
            
            # Lấy thông số từ metrics (video_level với pooling_strategy = "mean")
            video_metrics = metrics.get("video_level", {}).get("mean", {})
            
            accuracy = video_metrics.get("accuracy", 0)
            precision = video_metrics.get("precision", 0)
            recall = video_metrics.get("recall", 0)
            f1 = video_metrics.get("f1", 0)

            if return_code == 0:
                message = "Training completed successfully!"
                completed = True
                self._current_result_id = result_path
                print("✅", result_path)
            else:
                message = f"Training failed with code {return_code}"
                completed = False

            return {
                "message": message,
                "completed": completed,
                "model_path": result_path,  # ← THÊM DÒNG NÀY
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

        except Exception as e:
            self._is_running = False
            return {"error": str(e), "completed": False}

    def stop_training(self):
        """Dừng training process"""
        self._is_running = False
        if self._current_process:
            print("Stopping training process...")
            self._current_process.terminate()
            try:
                self._current_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._current_process.kill()
                self._current_process.wait()
            
            self._current_process = None
            return {"message": "Training stopped successfully!"}
        return {"message": "No training process is running!"}

    def save_model(self):
        """Lưu kết quả training vào database"""
        try:
            if not hasattr(self, '_current_result_id') or not self._current_result_id:
                return {"error": "No training result to save! Please run training first."}

            model_path = self._current_result_id
        
            # CHUYỂN ĐƯỜNG DẪN THÀNH ABSOLUTE VÀ CHUẨN HÓA
            if not os.path.isabs(model_path):
                # Nếu là relative path, chuyển thành absolute từ thư mục app
                app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                model_path = os.path.join(app_dir, model_path)
            
            # CHUẨN HÓA ĐƯỜNG DẪN (thay \ thành /)
            model_path = os.path.normpath(model_path)
            
            print(f"🔍 Checking model path: {model_path}")  # DEBUG
            
            # Kiểm tra đường dẫn tồn tại
            if not os.path.exists(model_path):
                return {"error": f"Model path not found: {model_path}. Current working dir: {os.getcwd()}"}

            metrics_file = os.path.join(model_path, "metrics.json")
            samples_file = os.path.join(model_path, "samples_list.xlsx")
            
            print(f"🔍 Checking metrics file: {metrics_file}")  # DEBUG
            print(f"🔍 Checking samples file: {samples_file}")  # DEBUG
            
            if not os.path.exists(metrics_file):
                return {"error": f"Metrics file not found: {metrics_file}"}
            
            if not os.path.exists(samples_file):
                return {"error": f"Samples list file not found: {samples_file}"}


            # Đọc metrics.json
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            # Đọc samples_list.xlsx
            df_samples = pd.read_excel(samples_file)
            
            # Lấy thông số từ metrics (video_level với pooling_strategy = "mean")
            video_metrics = metrics.get("video_level", {}).get("mean", {})
            
            accuracy = video_metrics.get("accuracy", 0)
            precision = video_metrics.get("precision", 0)
            recall = video_metrics.get("recall", 0)
            f1 = video_metrics.get("f1", 0)
            
            # Đường dẫn file model
            model_file_path = os.path.join(model_path, "model_final.h5")
            
            # Lưu từng sample vào database
            saved_count = 0
            for _, row in df_samples.iterrows():
                video_path = row['video_path']
                
                # Chuẩn hóa đường dẫn (thay \ thành /)
                normalized_path = video_path.replace('\\', '/')
                
                # Tách để lấy relative path (bỏ phần dataset/)
                if 'dataset/' in normalized_path:
                    relative_path = normalized_path.split('dataset/')[1]
                else:
                    relative_path = normalized_path
                
                # Lấy sample_id từ repository
                sample_repo = SampleRepository(self.db)
                sample = sample_repo.get_sample_by_file_path(relative_path)
                
                if sample:
                    # Lưu training result
                    training_repo = TrainingRepository(self.db)
                    training_result = training_repo.save_training_result(
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1=f1,
                        sample_id=sample.id,
                        user_id=None,  # hoặc lấy từ session nếu cần
                        model_path=model_file_path
                    )
                    
                    if training_result:
                        saved_count += 1
            
            return {
                "message": f"Model and {saved_count} training results saved successfully!",
                "saved_count": saved_count,
                "total_samples": len(df_samples)
            }

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Save model error: {error_details}")
            return {"error": f"Error saving model: {str(e)}"}

    def cancel_training(self):
        if self._current_result_id:
            self.repo.delete_training_result(self._current_result_id)
            self._current_result_id = None
        return {"message": "Training canceled."}
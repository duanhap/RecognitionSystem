import os
import subprocess
import sys
import asyncio
from typing import List
from app.repositories.training_repository import TrainingRepository
import random
import json
import pandas as pd
from app.repositories.sample_repository import SampleRepository
from app.model.training_result import TrainingResult
from app.core.config import settings
import signal
import threading
import time

class TrainingService:
    def __init__(self, db):
        self.db = db
        self.repo = TrainingRepository(db)
        self._current_result_id = None
        self._current_process = None
        self._is_running = False
        self._stop_event = threading.Event()

    async def start_training(self, train_type: str, num_samples: int, sample_mode: str, train_depth: str, train_mode: str, resume_model: str = None, user_id: int = None):
        print(f"Starting training: type={train_type}, samples={num_samples}, depth={train_depth}, mode={train_mode}, resume_model={resume_model}, user_id={user_id}")
        
        try:
            # Xác định file script theo loại
            app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
            # Xác định file script theo loại VÀ MODE
            if train_mode == "from_scratch":
                script_name = "train_video.py" if train_type == "video" else "train_image.py"
            else:  # resume mode
                script_name = "continue_train_video.py" if train_type == "video" else "continue_train_image.py"
            
            script_path = os.path.join(app_dir, script_name)

            if not os.path.exists(script_path):
                return {"error": f"Training script not found: {script_path}"}
            
            if train_mode == "from_scratch":
                # Xây dựng command
                cmd = [
                    sys.executable,
                    script_path,
                    "--dataset_root", f"dataset/{train_type}",
                    "--n_samples", str(num_samples),
                    "--sampling_mode", sample_mode,
                    "--depth", train_depth,
                    "--out_root", f"models/{train_type}"
                ]
            elif train_mode == "resume":
                if not resume_model:
                    return {"error": "Resume model path must be provided for resume training mode."}
                resume_model_full =  "models/video/"+resume_model+"/model_final.h5" if train_type == "video" else "models/image/"+resume_model+"/model_final.h5"
                print(f"🔍 Resume model full path: {resume_model_full}")
              
                cmd = [
                    sys.executable,
                    script_path,
                    "--model_path", resume_model_full,
                    "--n_samples", str(num_samples),
                    "--sampling_mode", sample_mode,
                    "--depth", train_depth,
                ]

            print("🚀 Running command:", " ".join(cmd))

            # Reset stop event
            self._stop_event.clear()
            
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
            result_path = None

            # Hàm đọc output real-time
            def read_output(pipe, pipe_name):
                nonlocal result_path
                try:
                    for line in pipe:
                        if self._stop_event.is_set():
                            break
                        line = line.strip()
                        if line:
                            print(f"{pipe_name}: {line}")
                            if "Results saved to:" in line:
                                result_path = line.split("Results saved to:")[-1].strip()
                            elif "Training finished! Results in:" in line:
                                result_path = line.split("Training finished! Results in:")[-1].strip()
                            elif "Training interrupted" in line or "interrupted" in line:
                                print("🛑 Training was interrupted")
                except Exception as e:
                    print(f"Error reading {pipe_name}: {e}")

            # Tạo threads để đọc stdout và stderr
            stdout_thread = threading.Thread(target=read_output, args=(self._current_process.stdout, "STDOUT"))
            stderr_thread = threading.Thread(target=read_output, args=(self._current_process.stderr, "STDERR"))
            
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            
            stdout_thread.start()
            stderr_thread.start()

            # Đợi process kết thúc hoặc bị stop
            while self._current_process.poll() is None and not self._stop_event.is_set():
                time.sleep(1)

            # Nếu stop event được set, dừng process
            if self._stop_event.is_set():
                print("🛑 Stop event detected, terminating process...")
                self._terminate_process()
                return {
                    "message": "Training stopped by user",
                    "completed": False,
                    "model_path": result_path
                }

            # Process đã kết thúc tự nhiên
            return_code = self._current_process.returncode
            self._current_process = None
            self._is_running = False

            # Xử lý kết quả sau khi training hoàn thành
            if return_code == 0 and result_path:
                # CHUYỂN ĐƯỜNG DẪN THÀNH ABSOLUTE VÀ CHUẨN HÓA
                if not os.path.isabs(result_path):
                    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                    result_path = os.path.join(app_dir, result_path)
                result_path = os.path.normpath(result_path)

                print(f"🔍 Checking model path: {result_path}")
                
                if not os.path.exists(result_path):
                    return {"error": f"Model path not found: {result_path}"}

                metrics_file = os.path.join(result_path, "metrics.json")
                print(f"🔍 Checking metrics file: {metrics_file}")
                
                if not os.path.exists(metrics_file):
                    return {"error": f"Metrics file not found: {metrics_file}"}

                # Đọc metrics.json
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
            
                if train_type == "image":
                    # Lấy thông số từ metrics (image level)
                    accuracy = metrics.get("accuracy", 0)
                    precision = (metrics.get("precision_real", 0)+metrics.get("precision_fake", 0))/2
                    recall = (metrics.get("recall_real", 0)+metrics.get("recall_fake", 0))/2
                    f1 = (metrics.get("f1_real", 0)+metrics.get("f1_fake", 0))/2
                    print(f"Image metrics - Acc: {accuracy}, Prec: {precision}, Recall: {recall}, F1: {f1}")
                else:
                    pooling_strategy = settings.VIDEO_TRAINING_CONFIG.get("default_pooling_strategy", "mean")
                    # Lấy thông số từ metrics (video_level với pooling_strategy = "mean")
                    video_metrics = metrics.get("video_level", {}).get(pooling_strategy, {})

                    accuracy = video_metrics.get("accuracy", 0)
                    precision = video_metrics.get("precision", 0)
                    recall = video_metrics.get("recall", 0)
                    f1 = video_metrics.get("f1", 0)
                    print(f"Video metrics - Acc: {accuracy}, Prec: {precision}, Recall: {recall}, F1: {f1}")

                message = "Training completed successfully!"
                completed = True
                self._current_result_id = result_path
                print("✅ Training completed:", result_path)

                return {
                    "message": message,
                    "completed": completed,
                    "model_path": result_path,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }
            else:
                return {
                    "message": f"Training failed with code {return_code}",
                    "completed": False,
                    "model_path": result_path
                }

        except Exception as e:
            self._is_running = False
            return {"error": str(e), "completed": False}

    def _terminate_process(self):
        """Dừng process training một cách an toàn"""
        if self._current_process:
            print("🛑 Terminating training process...")
            
            # Gửi SIGTERM trước (graceful shutdown)
            self._current_process.terminate()
            
            try:
                # Đợi process kết thúc trong 10 giây
                return_code = self._current_process.wait(timeout=10)
                print(f"✅ Process terminated with code: {return_code}")
            except subprocess.TimeoutExpired:
                print("⏰ Process not terminating, forcing kill...")
                # Nếu không terminate được, force kill
                self._current_process.kill()
                try:
                    self._current_process.wait(timeout=5)
                    print("✅ Process killed successfully")
                except subprocess.TimeoutExpired:
                    print("❌ Failed to kill process!")
            
            self._current_process = None
            self._is_running = False

    def stop_training(self):
        """Dừng training process từ bên ngoài"""
        print("🛑 Stop training requested...")
        self._stop_event.set()
        self._terminate_process()
        return {"message": "Training stop signal sent successfully!"}

    def save_model(self):
        """Lưu kết quả training vào database"""
        try:
            if not hasattr(self, '_current_result_id') or not self._current_result_id:
                return {"error": "No training result to save! Please run training first."}

            model_path = self._current_result_id
        
            # CHUYỂN ĐƯỜNG DẪN THÀNH ABSOLUTE VÀ CHUẨN HÓA
            if not os.path.isabs(model_path):
                app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                model_path = os.path.join(app_dir, model_path)
            
            model_path = os.path.normpath(model_path)
            
            print(f"🔍 Checking model path: {model_path}")
            
            if not os.path.exists(model_path):
                return {"error": f"Model path not found: {model_path}"}

            metrics_file = os.path.join(model_path, "metrics.json")
            samples_file = os.path.join(model_path, "samples_list.xlsx")
            
            print(f"🔍 Checking metrics file: {metrics_file}")
            print(f"🔍 Checking samples file: {samples_file}")
            
            if not os.path.exists(metrics_file):
                return {"error": f"Metrics file not found: {metrics_file}"}
            
            if not os.path.exists(samples_file):
                return {"error": f"Samples list file not found: {samples_file}"}

            # Đọc metrics.json
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            # Đọc samples_list.xlsx
            df_samples = pd.read_excel(samples_file)
            
            # Xác định loại model (video hay image)
            is_video_model = "video" in model_path
            
            if is_video_model:
                pooling_strategy = settings.VIDEO_TRAINING_CONFIG.get("default_pooling_strategy", "mean")
                video_metrics = metrics.get("video_level", {}).get(pooling_strategy, {})
                accuracy = video_metrics.get("accuracy", 0)
                precision = video_metrics.get("precision", 0)
                recall = video_metrics.get("recall", 0)
                f1 = video_metrics.get("f1", 0)
            else:        
                accuracy = metrics.get("accuracy", 0)
                precision = (metrics.get("precision_real", 0)+metrics.get("precision_fake", 0))/2
                recall = (metrics.get("recall_real", 0)+metrics.get("recall_fake", 0))/2
                f1 = (metrics.get("f1_real", 0)+metrics.get("f1_fake", 0))/2

            # Đường dẫn file model
            model_file_path = os.path.join(model_path, "model_final.h5")
            
            # Lưu từng sample vào database
            saved_count = 0
            for _, row in df_samples.iterrows():
                if is_video_model:
                    video_path = row['video_path']
                else:
                    video_path = row["filepath"]
                
                # Chuẩn hóa đường dẫn
                normalized_path = video_path.replace('\\', '/')
                
                # Tách để lấy relative path
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
                        user_id=None,
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
        """Hủy training và xóa kết quả"""
        return self.discard_training()

    def is_training_running(self):
        """Kiểm tra xem training có đang chạy không"""
        if self._current_process:
            # Kiểm tra process có còn sống không
            return self._current_process.poll() is None
        return False
    def discard_training(self):
        """Reset trạng thái training khi discard"""
        # Reset các biến quan trọng
        self._current_result_id = None
        self._is_running = False
        
        # Dừng process nếu đang chạy
        if self._current_process:
            self.stop_training()
        
        print("🗑️ Training state discarded - all variables reset")
        
        return {"message": "Training state discarded successfully!"}
    def get_history_results(self) -> List[TrainingResult]:
        return self.repo.list_results()
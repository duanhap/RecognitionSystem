# app/services/identity_training_service.py
import os
import subprocess
import sys
import threading
import time
import json
import pandas as pd
from typing import List
from app.repositories.training_repository import TrainingRepository
from app.repositories.sample_repository import SampleRepository
from app.model.training_result import TrainingResult
import numpy as np
from datetime import datetime

class IdentityTrainingService:
    def __init__(self, db):
        self.db = db
        self.training_repo = TrainingRepository(db)
        self.sample_repo = SampleRepository(db)
        self._current_model_path = None
        self._current_process = None
        self._is_running = False
        self._stop_event = threading.Event()

    async def start_training(self, train_mode: str, resume_model: str = None, user_id: int = None):
        print(f"Starting identity training: mode={train_mode}, resume_model={resume_model}")
        
        try:
            # Xác định script dựa trên mode
            if train_mode == "from_scratch":
                script_name = "train_identity_verification.py"
            else:  # continue mode
                script_name = "continue_training_identity.py"
            
            # Tìm đường dẫn script
            app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            script_path = os.path.join(app_dir, script_name)            
            
            if not os.path.exists(script_path):
                return {"error": f"Training script not found: {script_path}"}
            
            # Xây dựng command với đường dẫn đúng
            if train_mode == "from_scratch":
                cmd = [
                    sys.executable,
                    script_path,
                    "--dataset_root", os.path.join(app_dir, "dataset2"),
                    "--output_dir", os.path.join(app_dir, "models2")
                ]
            else:  # continue mode
                if not resume_model:
                    return {"error": "Resume model must be provided for continue training"}
                
                # Sửa đường dẫn resume model
                resume_model_path = os.path.join(app_dir, "models2", resume_model, "model_final.h5")
                if not os.path.exists(resume_model_path):
                    return {"error": f"Resume model not found: {resume_model_path}"}
                
                cmd = [
                    sys.executable,
                    script_path,
                    "--model_path", resume_model_path,
                    "--output_dir", os.path.join(app_dir, "models2")
                ]

            print("Running command:", " ".join(cmd))

            # Reset stop event
            self._stop_event.clear()
            
            # Chạy subprocess với working directory đúng
            self._current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=app_dir  # QUAN TRỌNG: set working directory
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
                            elif "Training completed! Results saved to:" in line:
                                result_path = line.split("Training completed! Results saved to:")[-1].strip()
                            elif "Continue training completed! Results saved to:" in line:
                                result_path = line.split("Continue training completed! Results saved to:")[-1].strip()
                except Exception as e:
                    print(f"Error reading {pipe_name}: {e}")

            # Tạo threads để đọc stdout và stderr
            stdout_thread = threading.Thread(target=read_output, args=(self._current_process.stdout, "STDOUT"))
            stderr_thread = threading.Thread(target=read_output, args=(self._current_process.stderr, "STDERR"))
            
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            
            stdout_thread.start()
            stderr_thread.start()

            # Đợi process kết thúc
            while self._current_process.poll() is None and not self._stop_event.is_set():
                time.sleep(1)

            # Nếu stop event được set
            if self._stop_event.is_set():
                print("Training stopped by user")
                self._terminate_process()
                return {
                    "message": "Training stopped by user",
                    "completed": False
                }

            # Process đã kết thúc tự nhiên
            return_code = self._current_process.returncode
            self._current_process = None
            self._is_running = False

            # Xử lý kết quả sau khi training hoàn thành
            if return_code == 0 and result_path:
                # Lưu model path - CHUYỂN THÀNH ABSOLUTE PATH
                if not os.path.isabs(result_path):
                    result_path = os.path.join(app_dir, result_path)
                
                self._current_model_path = result_path
                
                # Đọc metrics
                metrics_file = os.path.join(result_path, "metrics.json")
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                    
                    accuracy = metrics.get("accuracy", 0)
                    precision = metrics.get("precision", 0)
                    recall = metrics.get("recall", 0)
                    f1 = metrics.get("f1_score", 0)
                    total_samples = metrics.get("total_samples", 0)
                    
                    return {
                        "message": "Training completed successfully!",
                        "completed": True,
                        "model_path": result_path,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "total_samples": total_samples
                    }
                else:
                    return {"error": f"Metrics file not found: {metrics_file}"}
            else:
                return {
                    "message": f"Training failed with code {return_code}",
                    "completed": False
                }

        except Exception as e:
            self._is_running = False
            return {"error": str(e), "completed": False}

    # Trong hàm save_training_results, thêm debug:
    def save_training_results(self):
        """Lưu kết quả training vào database với embedding"""
        try:
            print(f"🔍 Current model path: {self._current_model_path}")
            
            if not self._current_model_path:
                return {"error": "No training result to save! Please run training first."}

            model_path = self._current_model_path
            
            print(f"Saving training results from: {model_path}")
            
            # Kiểm tra xem thư mục model có tồn tại không
            if not os.path.exists(model_path):
                return {"error": f"Model directory not found: {model_path}"}
            
            # Đọc file samples_embeddings.xlsx
            samples_file = os.path.join(model_path, "samples_embeddings.xlsx")
            if not os.path.exists(samples_file):
                return {"error": f"Samples embeddings file not found: {samples_file}"}
            
            # Đọc metrics
            metrics_file = os.path.join(model_path, "metrics.json")
            if not os.path.exists(metrics_file):
                return {"error": f"Metrics file not found: {metrics_file}"}
            
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            accuracy = metrics.get("accuracy", 0)
            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)
            f1 = metrics.get("f1_score", 0)
            
            print(f"📊 Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
            
            # Đọc samples embeddings
            df_samples = pd.read_excel(samples_file)
            print(f"📁 Found {len(df_samples)} samples in embeddings file")
            
            # Lưu từng sample vào database
            saved_count = 0
            not_found_count = 0
            
            for index, row in df_samples.iterrows():
                file_path = row['file_path']
                embedding = row['embedding']
                
                # Tìm sample tương ứng - SỬA: chỉ lấy tên file để tìm
                filename = os.path.basename(file_path)
                sample = self.sample_repo.get_sample_by_file_path(filename)
                
                if sample:
                    try:
                        # Lưu training result với embedding
                        training_result = TrainingResult(
                            acc=accuracy,
                            pre=precision,
                            rec=recall,
                            f1=f1,
                            file_path=os.path.join(model_path, "model_final.h5"),
                            created_at=datetime.now(),
                            training_sample_id=sample.id,
                            embedding=str(embedding)  # Lưu embedding
                        )
                        self.db.add(training_result)
                        saved_count += 1
                        print(f"✅ Saved training result for sample {sample.id} - {filename}")
                    except Exception as e:
                        print(f"❌ Error saving training result for sample {sample.id}: {e}")
                        continue
                else:
                    not_found_count += 1
                    print(f"⚠️ Sample not found for file: {filename}")
            
            # Commit tất cả
            self.db.commit()
            
            print(f"📈 Save summary: {saved_count} saved, {not_found_count} not found")
            
            # Reset state sau khi save
            self._current_model_path = None
            
            return {
                "message": f"Training results saved successfully! {saved_count} samples with embeddings saved.",
                "saved_count": saved_count,
                "total_samples": len(df_samples),
                "not_found_count": not_found_count
            }

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Save training results error: {error_details}")
            self.db.rollback()
            return {"error": f"Error saving training results: {str(e)}"}
    def discard_training(self):
        """Reset trạng thái training"""
        self._current_model_path = None
        self._is_running = False
        
        # Dừng process nếu đang chạy
        if self._current_process:
            self._terminate_process()
        
        return {"message": "Training state discarded successfully!"}

    def _terminate_process(self):
        """Dừng process training"""
        if self._current_process:
            self._current_process.terminate()
            try:
                self._current_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._current_process.kill()
            self._current_process = None
            self._is_running = False
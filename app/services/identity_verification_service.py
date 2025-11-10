# identity_verification_service.py
import uuid
import os
import json
from datetime import datetime
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

from app.core.model_loader import model_loader
from app.core.config import settings
from app.core.database import get_db
from app.repositories.training_repository import TrainingRepository
from app.model.training_result import TrainingResult
from app.model.training_sample import TrainingSample
from app.model.identity import Identity
from app.utils.grad_cam_utils import make_gradcam_heatmap, save_heatmap_image
import logging

logger = logging.getLogger(__name__)

class IdentityVerificationService:
    def __init__(self):
        self.identity_model = None  # Model 1: Nhận diện danh tính (so sánh embedding)
        self.image_fake_detection_model = None  # Model 2a: Phát hiện real/fake ảnh
        self.video_fake_detection_model = None  # Model 2b: Phát hiện real/fake video
        self.embedding_model = None  # FaceNet để extract embedding
        self.embedding_db = None
        self.person_db = None
        self.threshold = 0.6  # Threshold cho face recognition
        self.db_session = None
        self.training_repo = None
        logger.info("🔄 IdentityVerificationService initialized")
    
    async def initialize(self):
        """Khởi tạo service một cách rõ ràng"""
        try:
            logger.info("🔄 Starting IdentityVerificationService initialization...")
            self.db_session = next(get_db())
            self.training_repo = TrainingRepository(self.db_session)
            
            await self.load_identity_system()
            logger.info("✅ IdentityVerificationService fully initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize IdentityVerificationService: {e}")
            raise
    
    async def load_identity_system(self):
        """Load cả 3 model theo config"""
        try:
            logger.info("🔄 Loading identity system with 3 models...")
            
            # MODEL 1: Load identity recognition model (nếu có)
            identity_model_path = settings.IDENTITY_MODEL_PATH
            logger.info(f"🔍 Looking for Identity model at: {identity_model_path}")
            logger.info(f"🔍 Path exists: {identity_model_path.exists()}")
            
            if identity_model_path.exists():
                self.identity_model = tf.keras.models.load_model(identity_model_path)
                logger.info("✅ Model 1 - Identity recognition loaded")
                logger.info(f"📁 Model path: {identity_model_path}")
            else:
                logger.warning("⚠️ Model 1 - Identity model not found, using FaceNet only")
            
            # MODEL 2a: Load fake detection model cho ảnh
            image_model_path = settings.IMAGE_MODEL_PATH
            logger.info(f"🔍 Looking for Image Fake Detection model at: {image_model_path}")
            logger.info(f"🔍 Path exists: {image_model_path.exists()}")
            
            if image_model_path.exists():
                self.image_fake_detection_model = tf.keras.models.load_model(image_model_path)
                logger.info("✅ Model 2a - Image Fake Detection loaded")
                logger.info(f"📁 Model path: {image_model_path}")
            else:
                logger.error("❌ Model 2a - Image Fake Detection model not found")
                return False
            
            # MODEL 2b: Load fake detection model cho video
            video_model_path = settings.VIDEO_MODEL_PATH
            logger.info(f"🔍 Looking for Video Fake Detection model at: {video_model_path}")
            logger.info(f"🔍 Path exists: {video_model_path.exists()}")
            
            if video_model_path.exists():
                self.video_fake_detection_model = tf.keras.models.load_model(video_model_path)
                logger.info("✅ Model 2b - Video Fake Detection loaded")
                logger.info(f"📁 Model path: {video_model_path}")
            else:
                logger.error("❌ Model 2b - Video Fake Detection model not found")
                return False
            
            # FaceNet for embedding extraction
            self.embedding_model = self._load_embedding_model()
            
            # Load embedding database
            await self._load_identity_database()
            
            logger.info("✅ All 3 models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading identity system: {e}")
            return False
    
    async def _load_identity_database(self):
        """Load database với query tối ưu"""
        try:
            from sqlalchemy.orm import joinedload
            
            logger.info("🔄 Loading identity database...")
            
            # Chỉ load các field thực sự cần thiết
            results = self.db_session.query(
                TrainingResult.id,
                TrainingResult.embedding,
                TrainingResult.training_sample_id,
                TrainingResult.file_path,
                TrainingResult.created_at,
                TrainingSample.id.label('sample_id'),
                TrainingSample.type,
                TrainingSample.label,
                TrainingSample.file_path.label('sample_file_path'),
                Identity.id.label('identity_id'),
                Identity.name
            )\
            .join(TrainingResult.sample)\
            .outerjoin(TrainingSample.identity)\
            .filter(TrainingResult.embedding.isnot(None))\
            .all()
            
            logger.info(f"🔍 SQL query returned {len(results)} records with embeddings")
            
            self.embedding_db = []
            self.person_db = []
            
            valid_count = 0
            invalid_count = 0
            
            for i, result in enumerate(results):
                try:
                    if result.embedding is None:
                        invalid_count += 1
                        continue
                        
                    # Parse embedding
                    if isinstance(result.embedding, str):
                        embedding_str = result.embedding.strip('[]')
                        numbers = [float(x.strip()) for x in embedding_str.split(',')]
                        embedding = np.array(numbers)
                    elif isinstance(result.embedding, (list, np.ndarray)):
                        embedding = np.array(result.embedding)
                    else:
                        logger.warning(f"⚠️ Unknown embedding type: {type(result.embedding)}")
                        invalid_count += 1
                        continue
                    
                    # Validate embedding
                    if len(embedding) == 512:
                        person_info = {
                            'identity_id': result.identity_id,
                            'name': result.name or f"Unknown_{result.sample_id}",
                            'training_sample_id': result.sample_id,
                            'training_result_id': result.id,
                            'file_path': result.sample_file_path or result.file_path,
                            'sample_type': result.type,
                            'label': result.label,
                            'embedding': embedding,
                            'created_at': result.created_at
                        }
                        
                        self.embedding_db.append(embedding)
                        self.person_db.append(person_info)
                        valid_count += 1
                        
                    else:
                        invalid_count += 1
                        logger.warning(f"⚠️ Invalid embedding dimension: {len(embedding)} (expected 512)")
                            
                except Exception as e:
                    invalid_count += 1
                    logger.warning(f"⚠️ Error parsing embedding record {i+1}: {e}")
                    continue
            
            logger.info(f"✅ FINISHED: Loaded {valid_count} valid embeddings, {invalid_count} invalid")
            logger.info(f"📊 Embedding DB size: {len(self.embedding_db)}, Person DB size: {len(self.person_db)}")
            
            if self.embedding_db:
                logger.info(f"🔍 First person sample: {self.person_db[0]['name']}")
            else:
                logger.warning("⚠️ No embeddings loaded from database!")
            
        except Exception as e:
            logger.error(f"❌ Error loading identity database: {e}")
            raise

    def _load_embedding_model(self):
        """Load model để extract embedding"""
        try:
            from keras_facenet import FaceNet
            embedder = FaceNet()
            logger.info("✅ FaceNet embedding model loaded")
            return embedder
        except Exception as e:
            logger.error(f"❌ Error loading FaceNet: {e}")
            return None
    
    async def _extract_embedding(self, file, file_type):
        """Extract embedding từ file input"""
        try:
            if self.embedding_model is None:
                return None
            
            # Lưu file tạm
            file_extension = file.filename.split('.')[-1]
            filename = f"temp_{uuid.uuid4()}.{file_extension}"
            
            if file_type == "image":
                file_path = settings.IMAGE_UPLOAD_DIR / filename
            else:
                file_path = settings.VIDEO_UPLOAD_DIR / filename
            
            # Save file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Extract embedding
            if file_type == "image":
                embedding = self._extract_image_embedding(file_path)
            else:
                embedding = self._extract_video_embedding(file_path)
            
            # Xóa file tạm
            try:
                os.remove(file_path)
            except:
                pass
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Embedding extraction error: {e}")
            return None
    
    def _extract_image_embedding(self, image_path):
        """Extract embedding từ image"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (160, 160))
            
            embeddings = self.embedding_model.embeddings([image_resized])
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"❌ Image embedding extraction error: {e}")
            return None
    
    def _extract_video_embedding(self, video_path):
        """Extract embedding từ video"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            frame_embeddings = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % settings.IDENTITY_CONFIG["frame_interval"] == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_embedding = self._extract_single_frame_embedding(frame_rgb)
                    if frame_embedding is not None:
                        frame_embeddings.append(frame_embedding)
                
                frame_count += 1
                if len(frame_embeddings) >= settings.IDENTITY_CONFIG["max_frames_per_video"]:
                    break
            
            cap.release()
            
            if not frame_embeddings:
                return None
            
            # Apply pooling
            frame_embeddings = np.array(frame_embeddings)
            if settings.IDENTITY_CONFIG["video_pooling"] == "mean":
                return np.mean(frame_embeddings, axis=0)
            elif settings.IDENTITY_CONFIG["video_pooling"] == "max":
                return np.max(frame_embeddings, axis=0)
            else:
                return np.mean(frame_embeddings, axis=0)
                
        except Exception as e:
            logger.error(f"❌ Video embedding extraction error: {e}")
            return None
    
    def _extract_single_frame_embedding(self, frame):
        """Extract embedding từ single frame"""
        try:
            frame_resized = cv2.resize(frame, (160, 160))
            embeddings = self.embedding_model.embeddings([frame_resized])
            return embeddings[0]
        except:
            return None
    
    def _find_best_match(self, query_embedding):
        """Tìm người phù hợp nhất trong database"""
        if self.embedding_db is None or query_embedding is None or len(self.embedding_db) == 0:
            return None, 0
        
        try:
            query_embedding = query_embedding.reshape(1, -1)
            db_embeddings = np.array(self.embedding_db)
            
            similarities = cosine_similarity(query_embedding, db_embeddings)[0]
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            logger.info(f"🔍 Best similarity: {best_similarity:.3f} with threshold: {self.threshold}")
            
            if best_similarity >= self.threshold:
                best_person = self.person_db[best_idx]
                logger.info(f"✅ MATCH FOUND: {best_person['name']} with similarity {best_similarity:.3f}")
                return best_person, best_similarity
            else:
                logger.info(f"❌ NO MATCH: Best similarity {best_similarity:.3f} < threshold {self.threshold}")
                return None, best_similarity
                
        except Exception as e:
            logger.error(f"❌ Error finding best match: {e}")
            return None, 0
    
    def _get_person_info(self, person_data):
        """Lấy thông tin người từ person_data"""
        if person_data['identity_id']:
            return {
                "identity_id": person_data['identity_id'],
                "name": person_data['name'],
                "type": "verified_identity",
                "source": "database"
            }
        else:
            return {
                "identity_id": None,
                "name": person_data['name'],
                "type": "unknown_identity", 
                "source": "training_sample"
            }
    
    async def _run_fake_detection(self, file, file_type):
        """Chạy model fake detection (Model 2)"""
        try:
            if file_type == "image":
                from .image_service import ImageService
                service = ImageService()
                return await service.process_image(file)
            else:
                from .video_service import VideoService
                service = VideoService()
                return await service.process_video(file)
        except Exception as e:
            logger.error(f"❌ Fake detection error: {e}")
            return {
                "status": "error",
                "message": f"Fake detection failed: {str(e)}",
                "data": None,
                "timestamp": datetime.now().isoformat()
            }

    async def process_combined(self, file, file_type):
        """Process file với 2 model tuần tự theo logic yêu cầu"""
        temp_file_path = None
        try:
            # Đảm bảo service đã được khởi tạo
            if self.image_fake_detection_model is None or self.video_fake_detection_model is None:
                success = await self.load_identity_system()
                if not success:
                    return {
                        "status": "error",
                        "message": "Models not loaded properly",
                        "data": None,
                        "timestamp": datetime.now().isoformat()
                    }
            
            logger.info("🔄 Starting combined processing...")
            
            # LƯU FILE TẠM 1 LẦN DUY NHẤT để cả 2 service dùng
            file_extension = file.filename.split('.')[-1]
            filename = f"temp_{uuid.uuid4()}.{file_extension}"
            
            if file_type == "image":
                temp_file_path = settings.IMAGE_UPLOAD_DIR / filename
            else:
                temp_file_path = settings.VIDEO_UPLOAD_DIR / filename
            
            # Save file một lần
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Reset file pointer để có thể đọc lại
            await file.seek(0)
            
            # STEP 1: Model 1 - Nhận diện danh tính (so sánh embedding)
            query_embedding = await self._extract_embedding_from_path(temp_file_path, file_type)
            if query_embedding is None:
                return {
                    "status": "error", 
                    "message": "Cannot extract face embedding from file",
                    "data": None,
                    "timestamp": datetime.now().isoformat()
                }
            
            best_match, similarity = self._find_best_match(query_embedding)
            identity_found = best_match is not None
            
            # STEP 2: Model 2 - Fake detection (truyền đường dẫn file thay vì file object)
            fake_result = await self._run_fake_detection_from_path(temp_file_path, file_type)
            if fake_result["status"] == "error":
                return fake_result
            
            fake_data = fake_result["data"]
            is_fake = fake_data["label"] == "fake"
            fake_confidence = fake_data["confidence_score"]
            
            # STEP 3: Áp dụng logic business theo yêu cầu
            if identity_found and not is_fake:
                # ✅ Có trong DB + Real -> Hiện thông tin người
                conclusion = "real_verified"
                message = "✅ Người thật - Đã xác minh danh tính"
                risk_level = "low"
                person_info = self._get_person_info(best_match)
                
            elif identity_found and is_fake:
                # ⚠️ Có trong DB + Fake -> Giả mạo
                conclusion = "fake_impersonation"
                message = "⚠️ Cảnh báo giả mạo - Ảnh/video giả mạo người thật trong hệ thống"
                risk_level = "high" 
                person_info = self._get_person_info(best_match)
                
            elif not identity_found and not is_fake:
                # ❓ Không trong DB + Real -> Người lạ thật
                conclusion = "real_unknown" 
                message = "❓ Người lạ thật - Không có trong hệ thống"
                risk_level = "medium"
                person_info = None
                
            else:  # not identity_found and is_fake
                # 🚨 Không trong DB + Fake -> Deepfake nguy hiểm
                conclusion = "fake_unknown"
                message = "🚨 Cảnh báo giả mạo - Ảnh/video giả mạo người lạ không có trong hệ thống"
                risk_level = "critical"
                person_info = None
            
            # Build final result
            result_data = {
                "conclusion": conclusion,
                "message": message, 
                "risk_level": risk_level,
                "identity_verified": identity_found,
                "fake_detected": is_fake,
                "person_info": person_info,
                "similarity_score": float(similarity),
                "fake_confidence": fake_confidence,
                # "processing_details": {
                #     "model_1_identity_match": identity_found,
                #     "model_2_fake_detection": is_fake,
                #     "matched_person": best_match["name"] if identity_found else "Unknown",
                #     "similarity_threshold": self.threshold
                # }
            }
            
            # Kết hợp với fake detection data
            combined_data = {**fake_data, **result_data}
            
            return {
                "status": "success",
                "message": message,
                "data": combined_data,
                "timestamp": datetime.now().isoformat()
            }
                
        except Exception as e:
            logger.error(f"❌ Combined processing error: {e}")
            return {
                "status": "error",
                "message": f"Combined processing error: {str(e)}",
                "data": None,
                "timestamp": datetime.now().isoformat()
            }
        finally:
            # LUÔN xóa file tạm dù có lỗi hay không
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"🧹 Cleaned up temp file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete temp file: {e}")

    async def _extract_embedding_from_path(self, file_path, file_type):
        """Extract embedding từ file path (thay vì file object)"""
        try:
            if self.embedding_model is None:
                return None
            
            # Extract embedding từ file path
            if file_type == "image":
                embedding = self._extract_image_embedding(file_path)
            else:
                embedding = self._extract_video_embedding(file_path)
            
            return embedding
                
        except Exception as e:
            logger.error(f"❌ Embedding extraction error: {e}")
            return None

    async def _run_fake_detection_from_path(self, file_path, file_type):
        """Chạy model fake detection từ file path"""
        try:
            if file_type == "image":
                from .image_service import ImageService
                service = ImageService()
                # Gọi method mới từ ImageService để xử lý từ file path
                if hasattr(service, 'process_image_from_path'):
                    return await service.process_image_from_path(file_path)
                else:
                    # Fallback: mở file và gọi method cũ
                    with open(file_path, "rb") as f:
                        from fastapi import UploadFile
                        import io
                        file = UploadFile(filename=file_path.name, file=io.BytesIO(f.read()))
                        return await service.process_image(file)
            else:
                from .video_service import VideoService
                service = VideoService()
                if hasattr(service, 'process_video_from_path'):
                    return await service.process_video_from_path(file_path)
                else:
                    with open(file_path, "rb") as f:
                        from fastapi import UploadFile
                        import io
                        file = UploadFile(filename=file_path.name, file=io.BytesIO(f.read()))
                        return await service.process_video(file)
        except Exception as e:
            logger.error(f"❌ Fake detection error: {e}")
            return {
                "status": "error",
                "message": f"Fake detection failed: {str(e)}",
                "data": None,
                "timestamp": datetime.now().isoformat()
            }

    async def update_threshold(self, new_threshold):
        """Cập nhật threshold động"""
        old_threshold = self.threshold
        self.threshold = new_threshold
        logger.info(f"🔧 Threshold updated: {old_threshold} -> {new_threshold}")
        return {"old_threshold": old_threshold, "new_threshold": new_threshold}

    def get_service_status(self):
        """Lấy trạng thái service"""
        return {
            "model_1_identity_loaded": self.identity_model is not None,
            "model_2a_image_fake_detection_loaded": self.image_fake_detection_model is not None,
            "model_2b_video_fake_detection_loaded": self.video_fake_detection_model is not None,
            "embedding_model_loaded": self.embedding_model is not None,
            "database_loaded": self.embedding_db is not None,
            "embedding_count": len(self.embedding_db) if self.embedding_db else 0,
            "current_threshold": self.threshold,
            "model_paths": {
                "identity_model": str(settings.IDENTITY_MODEL_PATH),
                "image_fake_detection": str(settings.IMAGE_MODEL_PATH),
                "video_fake_detection": str(settings.VIDEO_MODEL_PATH)
            }
        }
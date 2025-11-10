import uuid
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
from PIL import Image
from ..core.model_loader import model_loader
from ..core.config import settings
from ..utils.grad_cam_utils import make_gradcam_heatmap, save_heatmap_image
import logging

logger = logging.getLogger(__name__)

class ImageService:
    def __init__(self):
        pass
    
    async def process_image(self, file):
        """Process uploaded image and return prediction results"""
        try:
            # Lưu file tạm và gọi process_image_from_path
            file_extension = file.filename.split('.')[-1]
            filename = f"{uuid.uuid4()}.{file_extension}"
            file_path = settings.IMAGE_UPLOAD_DIR / filename
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Gọi method xử lý từ path
            result = await self.process_image_from_path(file_path)
            
            # Xóa file tạm
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as cleanup_err:
                logger.warning(f"⚠️ Failed to delete temp file: {cleanup_err}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing image: {e}")
            return {
                "status": "error",
                "message": f"Error processing image: {str(e)}",
                "data": None,
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_image_from_path(self, image_path):
        """Process image từ file path thay vì UploadFile"""
        try:
            if model_loader.image_model is None:
                raise ValueError("Image model is not loaded. Please check model path.")
            
            # --- Xử lý ảnh từ path ---
            image = Image.open(image_path)

            # ĐẢM BẢO ẢNH CÓ 3 KÊNH RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.info(f"Converted image from {image.mode} to RGB")

            # KIỂM TRA KÍCH THƯỚC (debug)
            logger.info(f"Image size: {image.size}, mode: {image.mode}")

            image_array = np.array(image)
            logger.info(f"Image array shape: {image_array.shape}")

            label, confidence, prediction = model_loader.predict_image(image_array)
            
            # --- Sinh heatmap ---
            heatmap_filename = f"heatmap_{uuid.uuid4()}.jpg"
            heatmap_path = settings.IMAGE_HEATMAP_DIR / heatmap_filename
            
            img_for_heatmap = tf.image.resize(image_array, (299, 299))
            img_for_heatmap = tf.expand_dims(img_for_heatmap, axis=0) / 255.0
            
            pred_index = 1 if label == "fake" else 0
            heatmap = make_gradcam_heatmap(img_for_heatmap, model_loader.image_model, pred_index=pred_index)
            save_heatmap_image(image_path, heatmap, heatmap_path)
            
            # --- Trả kết quả ---
            result_data = {
                "heatmap_path": f"/static/heatmaps/images/{heatmap_filename}",
                "label": label,
                "confidence_score": round(confidence, 4),
                "raw_confidence": float(prediction[0][0]),  # Thêm raw confidence
                "created_at": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Image processed from path: {label} ({confidence:.2f})")
            
            return {
                "status": "success",
                "message": "Image processed successfully",
                "data": result_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing image from path: {e}")
            return {
                "status": "error",
                "message": f"Error processing image from path: {str(e)}",
                "data": None,
                "timestamp": datetime.now().isoformat()
            }
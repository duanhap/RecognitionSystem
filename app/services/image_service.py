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
            if model_loader.image_model is None:
                raise ValueError("Image model is not loaded. Please check model path.")
            
            # --- Lưu file tạm ---
            file_extension = file.filename.split('.')[-1]
            filename = f"{uuid.uuid4()}.{file_extension}"
            file_path = settings.IMAGE_UPLOAD_DIR / filename
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # --- Xử lý ảnh ---
            image = Image.open(file_path)
            image_array = np.array(image)
            
            label, confidence, prediction = model_loader.predict_image(image_array)
            
            # --- Sinh heatmap ---
            heatmap_filename = f"heatmap_{uuid.uuid4()}.jpg"
            heatmap_path = settings.IMAGE_HEATMAP_DIR / heatmap_filename
            
            img_for_heatmap = tf.image.resize(image_array, (299, 299))
            img_for_heatmap = tf.expand_dims(img_for_heatmap, axis=0) / 255.0
            
            pred_index = 1 if label == "fake" else 0
            heatmap = make_gradcam_heatmap(img_for_heatmap, model_loader.image_model, pred_index=pred_index)
            save_heatmap_image(file_path, heatmap, heatmap_path)
            
            # --- Dọn dẹp file tạm ---
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"🧹 Deleted temp image file: {file_path}")
            except Exception as cleanup_err:
                logger.warning(f"⚠️ Failed to delete temp file: {cleanup_err}")
            
            # --- Trả kết quả ---
            result = {
                "status": "success",
                "message": "Image processed successfully",
                "data": {
                    "heatmap_path": f"/static/heatmaps/images/{heatmap_filename}",
                    "label": label,
                    "confidence_score": round(confidence, 4),
                    "created_at": datetime.now().isoformat()
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Image processed: {filename} -> {label} ({confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing image: {e}")
            return {
                "status": "error",
                "message": f"Error processing image: {str(e)}",
                "data": None,
                "timestamp": datetime.now().isoformat()
            }

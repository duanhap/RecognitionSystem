import uuid
from datetime import datetime
import tensorflow as tf
import numpy as np
from PIL import Image
from ..core.model_loader import model_loader  # 👈 SỬA: import model_loader
from ..core.config import settings
from ..utils.grad_cam_utils import make_gradcam_heatmap, save_heatmap_image
import logging

logger = logging.getLogger(__name__)

class ImageService:
    def __init__(self):
        # 👈 SỬA: Sử dụng trực tiếp model_loader thay vì self.model
        pass
    
    async def process_image(self, file):
        """Process uploaded image and return prediction results"""
        try:
            # 👈 SỬA: Check model_loader thay vì self.model
            if model_loader.image_model is None:
                raise ValueError("Image model is not loaded. Please check model path.")
            
            # Generate unique filename
            file_extension = file.filename.split('.')[-1]
            filename = f"{uuid.uuid4()}.{file_extension}"
            file_path = settings.IMAGE_UPLOAD_DIR / filename
            
            # Save uploaded file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Load and preprocess image
            image = Image.open(file_path)
            image_array = np.array(image)
            
            # Predict using model_loader
            label, confidence, prediction = model_loader.predict_image(image_array)
            
            # Generate heatmap - 👈 SỬA: dùng model_loader.image_model
            heatmap_filename = f"heatmap_{uuid.uuid4()}.jpg"
            heatmap_path = settings.IMAGE_HEATMAP_DIR / heatmap_filename
            
            img_for_heatmap = tf.image.resize(image_array, (299, 299))
            img_for_heatmap = tf.expand_dims(img_for_heatmap, axis=0) / 255.0
            
            pred_index = 1 if label == "fake" else 0
            heatmap = make_gradcam_heatmap(img_for_heatmap, model_loader.image_model, pred_index=pred_index)  # 👈 SỬA
            save_heatmap_image(file_path, heatmap, heatmap_path)
            
            # Prepare response với format chuẩn
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
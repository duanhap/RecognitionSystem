import uuid
from datetime import datetime
import cv2
import tensorflow as tf  # 👈 THÊM import tensorflow
import numpy as np
from ..core.model_loader import model_loader  # 👈 SỬA
from ..core.config import settings
from ..utils.grad_cam_utils import make_gradcam_heatmap, save_heatmap_image
import logging

logger = logging.getLogger(__name__)

class VideoService:
    def __init__(self):
        # 👈 SỬA: Sử dụng trực tiếp model_loader
        pass
    
    async def process_video(self, file):
        """Process uploaded video and return prediction results"""
        try:
            # 👈 SỬA: Check model_loader
            if model_loader.video_model is None:
                raise ValueError("Video model is not loaded. Please check model path.")
            
            # Generate unique filename
            file_extension = file.filename.split('.')[-1]
            filename = f"{uuid.uuid4()}.{file_extension}"
            file_path = settings.VIDEO_UPLOAD_DIR / filename
            
            # Save uploaded file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Extract middle frame for analysis
            frame = self.extract_middle_frame(file_path)
            if frame is None:
                raise ValueError("Could not extract frame from video")
            
            # Predict using model_loader
            label, confidence, prediction = model_loader.predict_image(frame)
            
            # Generate heatmap for the frame
            heatmap_filename = f"heatmap_{uuid.uuid4()}.jpg"
            heatmap_path = settings.VIDEO_HEATMAP_DIR / heatmap_filename
            
            # Save frame temporarily for heatmap
            frame_path = settings.VIDEO_UPLOAD_DIR / f"temp_frame_{uuid.uuid4()}.jpg"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            frame_for_heatmap = tf.image.resize(frame, (299, 299))
            frame_for_heatmap = tf.expand_dims(frame_for_heatmap, axis=0) / 255.0
            
            pred_index = 1 if label == "fake" else 0
            heatmap = make_gradcam_heatmap(frame_for_heatmap, model_loader.video_model, pred_index=pred_index)  # 👈 SỬA
            save_heatmap_image(frame_path, heatmap, heatmap_path)
            
            # Cleanup temp frame
            frame_path.unlink(missing_ok=True)
            
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
            
            logger.info(f"✅ Video processed: {filename} -> {label} ({confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing image: {e}")
            return {
                "status": "error",
                "message": f"Error processing image: {str(e)}",
                "data": None,
                "timestamp": datetime.now().isoformat()
            }
    def extract_middle_frame(self, video_path):
        """Extract middle frame from video"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return None
            
        middle_frame_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
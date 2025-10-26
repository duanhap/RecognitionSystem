import uuid
from datetime import datetime
import cv2
import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from ..core.model_loader import model_loader
from ..core.config import settings
from ..utils.grad_cam_utils import make_gradcam_heatmap, save_heatmap_image
import logging

logger = logging.getLogger(__name__)

class VideoService:
    def __init__(self):
        self.frame_interval = 10
        self.max_frames = 30  # Giới hạn số frame để xử lý nhanh
        self.pooling_strategy = getattr(settings, 'VIDEO_POOLING_STRATEGY', 'mean')  # 👈 
    
    async def process_video(self, file):
        """Process uploaded video with multiple frames and video-level pooling"""
        try:
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
            
            # Extract multiple frames for video-level analysis
            frames_data = self.extract_frames(file_path)
            if not frames_data:
                raise ValueError("Could not extract frames from video")
            
            # Predict using multiple frames với video-level pooling
            video_result = self.predict_video_with_pooling(frames_data)
            
            # Generate heatmap for frame with highest confidence
            best_frame_data = self.get_best_frame(frames_data, video_result["frame_predictions"])
            heatmap_filename = f"heatmap_{uuid.uuid4()}.jpg"
            heatmap_path = settings.VIDEO_HEATMAP_DIR / heatmap_filename
            
            self.generate_heatmap_for_frame(best_frame_data, heatmap_path, video_result["final_label"])
            
            # Cleanup temp files
            self.cleanup_temp_frames(frames_data)
            file_path.unlink(missing_ok=True)  # Xóa video gốc sau khi xử lý
            
            # Prepare response với video-level results
            result = {
                "status": "success",
                "message": "Video processed successfully with video-level analysis",
                "data": {
                    "heatmap_path": f"/static/heatmaps/videos/{heatmap_filename}",
                    "label": video_result["final_label"],
                    "confidence_score": round(video_result["final_confidence"], 4),
                    "pooling_strategy": video_result["pooling_strategy"],
                    "frames_analyzed": video_result["frames_analyzed"],
                    "frame_predictions": video_result["frame_predictions"],
                    "pooling_results": video_result["pooling_results"],  # 👈 THÊM DÒNG NÀY
                    "created_at": datetime.now().isoformat()
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Video processed: {filename} -> {video_result['final_label']} "
                       f"({video_result['final_confidence']:.2f}) using {video_result['frames_analyzed']} frames")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing video: {e}")
            # Cleanup on error
            try:
                file_path.unlink(missing_ok=True)
                self.cleanup_temp_frames(frames_data if 'frames_data' in locals() else [])
            except:
                pass
            
            return {
                "status": "error",
                "message": f"Error processing video: {str(e)}",
                "data": None,
                "timestamp": datetime.now().isoformat()
            }
    
    def extract_frames(self, video_path):
        """Extract multiple frames from video với interval"""
        frames_data = []
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return frames_data
        
        frame_idx = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.frame_interval == 0:
                # Convert and resize frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, model_loader.img_size)
                
                # Save frame temporarily for heatmap
                frame_filename = f"temp_frame_{uuid.uuid4()}.jpg"
                frame_path = settings.VIDEO_UPLOAD_DIR / frame_filename
                cv2.imwrite(str(frame_path), frame_rgb)  # Save original for heatmap
                
                frames_data.append({
                    "frame_array": frame_resized,
                    "frame_path": frame_path,
                    "frame_index": frame_idx
                })
                extracted_count += 1
            
            frame_idx += 1
        
        cap.release()
        return frames_data
    
    def predict_video_with_pooling(self, frames_data):
        """Predict video using multiple frames với pooling strategies"""
        frame_predictions = []
        frame_confidences = []
        
        for frame_data in frames_data:
            # Predict single frame
            label, confidence, prediction = model_loader.predict_video_frame(frame_data["frame_array"])
            
            frame_predictions.append({
                "label": label,
                "confidence": confidence,
                "fake_probability": float(prediction[0][1]),  # Probability of being fake
                "frame_index": frame_data["frame_index"]
            })
            frame_confidences.append(confidence)
        
        # Apply video-level pooling (giống training pipeline)
        fake_probabilities = [pred["fake_probability"] for pred in frame_predictions]
        fake_probs_array = np.array(fake_probabilities)
        
         # Tính toán pooling results với xử lý lỗi
        pooling_results = {
            "mean": np.mean(fake_probs_array),
            "max": np.max(fake_probs_array),
            "median": np.median(fake_probs_array),
            "q75": np.quantile(fake_probs_array, 0.75),
        }
        
        # Thêm confidence_weighted với xử lý đặc biệt
        try:
            weights = np.abs(fake_probs_array - 0.5)
            # Tránh chia cho 0 nếu tất cả weights = 0
            if np.sum(weights) > 0:
                pooling_results["confidence_weighted"] = np.average(fake_probs_array, weights=weights)
            else:
                pooling_results["confidence_weighted"] = pooling_results["mean"]
        except Exception as e:
            logger.warning(f"Error in confidence_weighted pooling: {e}, using mean instead")
            pooling_results["confidence_weighted"] = pooling_results["mean"]
        
        
        # 🎯 SỬA: Sử dụng pooling strategy từ config (hoặc default)
        pooling_strategy = getattr(self, 'pooling_strategy', 'mean')
        final_fake_prob = pooling_results[pooling_strategy]
        final_label = "fake" if final_fake_prob > 0.5 else "real"
        final_confidence = final_fake_prob if final_label == "fake" else 1 - final_fake_prob
        
        return {
            "final_label": final_label,
            "final_confidence": final_confidence,
            "pooling_strategy": pooling_strategy,  # Trả về strategy thực tế đã dùng
            "frames_analyzed": len(frames_data),
            "frame_predictions": frame_predictions,
            "pooling_results": pooling_results
        }
    
    def get_best_frame(self, frames_data, frame_predictions):
        """Get frame with highest confidence for heatmap generation"""
        best_idx = 0
        best_confidence = 0
        
        for i, pred in enumerate(frame_predictions):
            if pred["confidence"] > best_confidence:
                best_confidence = pred["confidence"]
                best_idx = i
        
        return frames_data[best_idx]
    
    def generate_heatmap_for_frame(self, frame_data, heatmap_path, predicted_label):
        """Generate Grad-CAM heatmap for the best frame"""
        try:
            frame_array = frame_data["frame_array"]
            frame_path = frame_data["frame_path"]
            
            # Preprocess frame for model
            frame_for_heatmap = tf.image.resize(frame_array, model_loader.img_size)
            frame_for_heatmap = tf.expand_dims(frame_for_heatmap, axis=0) / 255.0
            
            # Determine prediction index for heatmap
            pred_index = 1 if predicted_label == "fake" else 0
            
            # Generate heatmap
            heatmap = make_gradcam_heatmap(
                frame_for_heatmap, 
                model_loader.video_model, 
                pred_index=pred_index
            )
            
            # Save heatmap
            save_heatmap_image(frame_path, heatmap, heatmap_path)
            
        except Exception as e:
            logger.error(f"❌ Error generating heatmap: {e}")
            raise
    
    def cleanup_temp_frames(self, frames_data):
        """Cleanup temporary frame files"""
        for frame_data in frames_data:
            try:
                frame_data["frame_path"].unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Could not delete temp frame: {e}")
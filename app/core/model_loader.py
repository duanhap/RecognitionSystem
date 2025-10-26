import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from .config import settings
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.image_model = None
        self.video_model = None
        self.img_size = (299, 299)
        
    def load_models(self):
        """Load image and video models"""
        try:
            # Load image model
            if settings.IMAGE_MODEL_PATH.exists():
                self.image_model = load_model(settings.IMAGE_MODEL_PATH)
                logger.info("✅ Image model loaded successfully")
            else:
                logger.error(f"❌ Image model not found at {settings.IMAGE_MODEL_PATH}")
            
            # Load video model
            if settings.VIDEO_MODEL_PATH.exists():
                self.video_model = load_model(settings.VIDEO_MODEL_PATH)
                logger.info("✅ Video model loaded successfully")
            else:
                logger.error(f"❌ Video model not found at {settings.VIDEO_MODEL_PATH}")
                
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
    
    def predict_image(self, image_array):
        """Predict single image"""
        if self.image_model is None:
            raise ValueError("Image model not loaded")
        
        # Preprocess image
        img = tf.image.resize(image_array, self.img_size)
        img = tf.expand_dims(img, axis=0)
        img = img / 255.0
        
        # Predict
        prediction = self.image_model.predict(img, verbose=0)
        confidence = float(np.max(prediction))
        label = "fake" if np.argmax(prediction) == 1 else "real"
        
        return label, confidence, prediction
    
    def predict_video_frame(self, frame_array):
        """Predict single video frame - tương thích với video model"""
        if self.video_model is None:
            raise ValueError("Video model not loaded")
        
        # Preprocess frame (giống trong training)
        img = tf.image.resize(frame_array, self.img_size)
        img = tf.expand_dims(img, axis=0)
        img = img / 255.0
        
        # Predict
        prediction = self.video_model.predict(img, verbose=0)
        confidence = float(np.max(prediction))
        label = "fake" if np.argmax(prediction) == 1 else "real"
        
        return label, confidence, prediction

model_loader = ModelLoader()
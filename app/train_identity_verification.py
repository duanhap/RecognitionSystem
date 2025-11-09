# train_identity_verification.py
import os
import cv2
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from keras_facenet import FaceNet
import argparse

# Import config
from core.config import settings

# ==========================
# ⚙️ CONFIG IDENTITY VERIFICATION
# ==========================
IDENTITY_CONFIG = {
    "model_type": "facenet",
    "embedding_dim": 512,
    "img_size": (160, 160),
    "frame_interval": 10,
    "max_frames_per_video": 10,
    "video_pooling": "mean",
    "train_ratio": 0.8,
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 1e-4,
    "patience": 5,
    "default_dataset_root": "dataset2"
}

# ==========================
# 🧠 FACE DETECTION & EMBEDDING
# ==========================
class IdentityVerificationSystem:
    def __init__(self):
        if IDENTITY_CONFIG["model_type"] == "facenet":
            self.embedder = FaceNet()
            self.embedding_dim = 512
        else:
            self.base_model = Xception(weights="imagenet", include_top=False, 
                                     input_shape=(*IDENTITY_CONFIG["img_size"], 3))
            self.embedding_dim = 2048
            
    def get_embedding(self, image):
        """Extract embedding from image"""
        if IDENTITY_CONFIG["model_type"] == "facenet":
            image_resized = cv2.resize(image, IDENTITY_CONFIG["img_size"])
            embeddings = self.embedder.embeddings([image_resized])
            return embeddings[0]
        else:
            image_resized = cv2.resize(image, IDENTITY_CONFIG["img_size"])
            image_array = image_resized / 255.0
            image_batch = np.expand_dims(image_array, axis=0)
            features = self.base_model.predict(image_batch, verbose=0)
            return GlobalAveragePooling2D()(features).numpy().flatten()

# ==========================
# 🎬 VIDEO PROCESSING
# ==========================
def extract_video_embedding(video_path, system):
    """Extract single embedding from video using pooling"""
    cap = cv2.VideoCapture(video_path)
    frame_embeddings = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % IDENTITY_CONFIG["frame_interval"] == 0:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                embedding = system.get_embedding(frame_rgb)
                frame_embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                
        frame_count += 1
        
        if len(frame_embeddings) >= IDENTITY_CONFIG["max_frames_per_video"]:
            break
            
    cap.release()
    
    if not frame_embeddings:
        return None
        
    frame_embeddings = np.array(frame_embeddings)
    
    if IDENTITY_CONFIG["video_pooling"] == "mean":
        return np.mean(frame_embeddings, axis=0)
    elif IDENTITY_CONFIG["video_pooling"] == "max":
        return np.max(frame_embeddings, axis=0)
    elif IDENTITY_CONFIG["video_pooling"] == "median":
        return np.median(frame_embeddings, axis=0)
    else:
        return np.mean(frame_embeddings, axis=0)

# ==========================
# 📁 DATASET PROCESSING
# ==========================
# train_identity_verification.py - SỬA PHẦN DATASET PROCESSING

# SỬA PHẦN PROCESS PHOTOS - THÊM ĐIỀU KIỆN KIỂM TRA ẢNH

def process_dataset(dataset_root, system):
    """Process both photos and videos from dataset"""
    embeddings = []
    labels = []
    file_paths = []
    file_types = []
    
    # Process photos
    photo_dir = os.path.join(dataset_root, "image")
    if os.path.exists(photo_dir):
        print("Processing photos...")
        for person_name in tqdm(os.listdir(photo_dir)):
            person_dir = os.path.join(photo_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            for img_file in os.listdir(person_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, img_file)
                    try:
                        img = cv2.imread(img_path)
                        # THÊM KIỂM TRA ẢNH CÓ LOAD ĐƯỢC KHÔNG
                        if img is not None and img.size > 0:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            embedding = system.get_embedding(img_rgb)
                            embeddings.append(embedding)
                            labels.append(person_name)
                            file_paths.append(img_path)
                            file_types.append("photo")
                            print(f"✅ Processed photo: {img_path}")
                        else:
                            print(f"❌ Failed to load image: {img_path}")
                    except Exception as e:
                        print(f"❌ Error processing {img_path}: {e}")
    
    # Process videos
    video_dir = os.path.join(dataset_root, "video")
    if os.path.exists(video_dir):
        print("Processing videos...")
        for person_name in tqdm(os.listdir(video_dir)):
            person_dir = os.path.join(video_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            for video_file in os.listdir(person_dir):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(person_dir, video_file)
                    try:
                        embedding = extract_video_embedding(video_path, system)
                        if embedding is not None:
                            embeddings.append(embedding)
                            labels.append(person_name)
                            file_paths.append(video_path)
                            file_types.append("video")
                            print(f"✅ Processed video: {video_path}")
                        else:
                            print(f"❌ Failed to process video: {video_path}")
                    except Exception as e:
                        print(f"❌ Error processing {video_path}: {e}")
    
    # THÊM IN RA THỐNG KÊ
    print(f"\n📊 FINAL STATS:")
    print(f"Total samples: {len(embeddings)}")
    print(f"Photos: {len([t for t in file_types if t == 'photo'])}")
    print(f"Videos: {len([t for t in file_types if t == 'video'])}")
    print(f"Persons: {len(set(labels))}")
    
    return np.array(embeddings), np.array(labels), np.array(file_paths), np.array(file_types)
# ==========================
# 🧠 MODEL TRAINING
# ==========================
def build_classification_model(input_dim, num_classes):
    """Build classifier on top of embeddings"""
    model = tf.keras.Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=IDENTITY_CONFIG["learning_rate"]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ==========================
# 🚀 MAIN TRAINING PIPELINE
# ==========================
def train_identity_verification(dataset_root=None, output_dir=None):
    """Main training function"""
    # Setup paths
    dataset_root = dataset_root or IDENTITY_CONFIG["default_dataset_root"]
    output_dir = output_dir or "models2"
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = os.path.join(output_dir, f"{timestamp}_identity_verification")
    os.makedirs(model_folder, exist_ok=True)
    
    print("🚀 Starting Identity Verification Training...")
    
    # Initialize system
    system = IdentityVerificationSystem()
    
    # Process dataset
    print("📁 Processing dataset...")
    X, y, file_paths, file_types = process_dataset(dataset_root, system)
    
    if len(X) == 0:
        raise ValueError("No data processed from dataset!")
    
    print(f"✅ Processed {len(X)} samples ({len(np.unique(y))} persons)")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    # Split data
    X_train, X_test, y_train, y_test, paths_train, paths_test, types_train, types_test = train_test_split(
        X, y_encoded, file_paths, file_types, 
        train_size=IDENTITY_CONFIG["train_ratio"], 
        random_state=42, 
        #stratify=y_encoded
    )
    
    # Build and train model
    print("🧠 Training classifier...")
    model = build_classification_model(X.shape[1], num_classes)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=IDENTITY_CONFIG["patience"], restore_best_weights=True),
        ModelCheckpoint(os.path.join(model_folder, "model_checkpoint.h5"), save_best_only=True),
        CSVLogger(os.path.join(model_folder, "training_log.csv"))
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=IDENTITY_CONFIG["epochs"],
        batch_size=IDENTITY_CONFIG["batch_size"],
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(model_folder, "model_final.h5"))
    
    # Evaluation
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_classes, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_classes, average='weighted', zero_division=0)
    
    # Save results
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "num_classes": num_classes,
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "config": IDENTITY_CONFIG
    }
    
    with open(os.path.join(model_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save samples and embeddings
    samples_data = []
    for i in range(len(X)):
        samples_data.append({
            "file_path": file_paths[i],
            "person_id": y[i],
            "file_type": file_types[i],
            "embedding": X[i].tolist()
        })
    
    df_samples = pd.DataFrame(samples_data)
    df_samples.to_excel(os.path.join(model_folder, "samples_embeddings.xlsx"), index=False)
    
    # Save test results
    test_results = []
    for i in range(len(X_test)):
        test_results.append({
            "file_path": paths_test[i],
            "true_person": le.inverse_transform([y_test[i]])[0],
            "predicted_person": le.inverse_transform([y_pred_classes[i]])[0],
            "file_type": types_test[i],
            "correct": y_test[i] == y_pred_classes[i]
        })
    
    df_test_results = pd.DataFrame(test_results)
    df_test_results.to_excel(os.path.join(model_folder, "test_results.xlsx"), index=False)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, "training_curves.png"))
    plt.close()
    
    print(f"✅ Training completed! Results saved to: {model_folder}")
    print(f"📊 Final Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return model_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Identity Verification System")
    parser.add_argument("--dataset_root", type=str, default=None, help="Path to dataset2 directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: models2)")
    
    args = parser.parse_args()
    train_identity_verification(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir
    )
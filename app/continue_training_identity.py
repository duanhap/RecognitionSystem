# continue_training_identity.py
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
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Import existing modules
from train_identity_verification import (
    IdentityVerificationSystem, 
    extract_video_embedding,
    build_classification_model,
    IDENTITY_CONFIG
)

# continue_training_identity.py - SỬA PHẦN PROCESS NEW DATASET

def process_new_dataset(dataset_root, system, existing_files):
    """Process only NEW files that don't exist in current database - STRUCTURE BY PERSON NAME"""
    embeddings = []
    labels = []
    file_paths = []
    file_types = []
    
    # Process photos - STRUCTURE: dataset2/image/Person Name/*.jpg
    photo_dir = os.path.join(dataset_root, "image")
    if os.path.exists(photo_dir):
        print("Checking for NEW photos...")
        for person_name in tqdm(os.listdir(photo_dir)):
            person_dir = os.path.join(photo_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            for img_file in os.listdir(person_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, img_file)
                    
                    # 🔍 CHECK IF FILE ALREADY EXISTS
                    if img_path in existing_files:
                        continue  # Skip existing files
                        
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            embedding = system.get_embedding(img_rgb)
                            embeddings.append(embedding)
                            labels.append(person_name)  # Use person name as label
                            file_paths.append(img_path)
                            file_types.append("photo")
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
    
    # Process videos - STRUCTURE: dataset2/video/Person Name/*.mp4
    video_dir = os.path.join(dataset_root, "video")
    if os.path.exists(video_dir):
        print("Checking for NEW videos...")
        for person_name in tqdm(os.listdir(video_dir)):
            person_dir = os.path.join(video_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            for video_file in os.listdir(person_dir):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(person_dir, video_file)
                    
                    # 🔍 CHECK IF FILE ALREADY EXISTS
                    if video_path in existing_files:
                        continue  # Skip existing files
                        
                    try:
                        embedding = extract_video_embedding(video_path, system)
                        if embedding is not None:
                            embeddings.append(embedding)
                            labels.append(person_name)  # Use person name as label
                            file_paths.append(video_path)
                            file_types.append("video")
                    except Exception as e:
                        print(f"Error processing {video_path}: {e}")
    
    return np.array(embeddings), np.array(labels), np.array(file_paths), np.array(file_types)
def continue_identity_training(model_path, new_data_root=None, output_dir=None):
    """Continue training with existing model - only process NEW data"""
    # Setup paths
    new_data_root = new_data_root or IDENTITY_CONFIG["default_dataset_root"]
    output_dir = output_dir or "models2"
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_model_name = os.path.basename(os.path.dirname(model_path))
    model_folder = os.path.join(output_dir, f"{timestamp}_continue_{base_model_name}")
    os.makedirs(model_folder, exist_ok=True)
    
    print("🔄 Continuing Identity Verification Training...")
    
    # Load existing model and data
    print(f"📂 Loading existing model: {model_path}")
    model = load_model(model_path)
    
    # Load existing samples data
    existing_model_dir = os.path.dirname(model_path)
    existing_samples_path = os.path.join(existing_model_dir, "samples_embeddings.xlsx")
    
    if os.path.exists(existing_samples_path):
        df_existing = pd.read_excel(existing_samples_path)
        existing_files = set(df_existing['file_path'].values)
        print(f"📊 Loaded {len(existing_files)} existing samples")
        
        # Convert embeddings from existing data
        X_existing = np.array([np.fromstring(emb.strip('[]'), sep=',') for emb in df_existing['embedding']])
        y_existing = df_existing['person_id'].values
        paths_existing = df_existing['file_path'].values
        types_existing = df_existing['file_type'].values
    else:
        print("⚠️ No existing samples found, starting from scratch")
        existing_files = set()
        X_existing = np.array([])
        y_existing = np.array([])
        paths_existing = np.array([])
        types_existing = np.array([])
    
    # Process ONLY NEW data
    print("📁 Processing NEW dataset...")
    system = IdentityVerificationSystem()
    X_new, y_new, paths_new, types_new = process_new_dataset(new_data_root, system, existing_files)
    
    if len(X_new) == 0:
        print("🎉 No NEW data found! Model is already up-to-date.")
        return None
    
    print(f"✅ Found {len(X_new)} NEW samples to train")
    
    # Combine existing data + new data
    if len(X_existing) > 0:
        X_combined = np.vstack([X_existing, X_new])
        y_combined = np.concatenate([y_existing, y_new])
        paths_combined = np.concatenate([paths_existing, paths_new])
        types_combined = np.concatenate([types_existing, types_new])
    else:
        X_combined = X_new
        y_combined = y_new
        paths_combined = paths_new
        types_combined = types_new
    
    print(f"📦 Combined dataset: {len(X_combined)} total samples ({len(X_existing)} existing + {len(X_new)} new)")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_combined)
    num_classes = len(le.classes_)
    
    # Rebuild model if number of classes changed
    if model.output.shape[-1] != num_classes:
        print(f"🔄 Number of classes changed from {model.output.shape[-1]} to {num_classes}, rebuilding model...")
        model = build_classification_model(X_combined.shape[1], num_classes)
    else:
        # Continue with existing model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=IDENTITY_CONFIG["learning_rate"]),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # Split data
    X_train, X_test, y_train, y_test, paths_train, paths_test, types_train, types_test = train_test_split(
        X_combined, y_encoded, paths_combined, types_combined,
        train_size=IDENTITY_CONFIG["train_ratio"], 
        random_state=42, 
        #stratify=y_encoded
    )
    
    # Continue training
    print("🧠 Continuing training...")
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
        "total_samples": len(X_combined),
        "existing_samples": len(X_existing),
        "new_samples": len(X_new),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "config": IDENTITY_CONFIG
    }
    
    with open(os.path.join(model_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save COMBINED samples and embeddings (existing + new)
    samples_data = []
    for i in range(len(X_combined)):
        samples_data.append({
            "file_path": paths_combined[i],
            "person_id": y_combined[i],
            "file_type": types_combined[i],
            "embedding": X_combined[i].tolist(),
            "data_source": "existing" if i < len(X_existing) else "new"
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
            "data_source": "existing" if paths_test[i] in paths_existing else "new",
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
    
    print(f"✅ Continue training completed! Results saved to: {model_folder}")
    print(f"📊 Final Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"📁 Dataset: {len(X_existing)} existing + {len(X_new)} new = {len(X_combined)} total samples")
    
    return model_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue Identity Verification Training")
    parser.add_argument("--model_path", type=str, required=True, help="Path to existing model")
    parser.add_argument("--new_data_root", type=str, default=None, help="Path to new dataset")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    continue_identity_training(args.model_path, args.new_data_root, args.output_dir)
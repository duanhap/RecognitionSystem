"""
continue_train.py

Tiếp tục training từ model đã có sẵn với các cải tiến:
- Sửa lỗi Grad-CAM (RGBA -> RGB)
- Cân bằng real/fake
- Đúng kích thước Xception (299, 299)
- Có validation set
- Đầu ra đầy đủ như train_image.py
"""

import os
import json
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import seaborn as sns
import cv2
import tensorflow as tf
from PIL import Image
from glob import glob

# ========== CONFIG ==========
DATASET_DIR = "dataset/image"
MODEL_DIR = "models/image"
IMG_SIZE = (299, 299)  # Xception requirement

# DEPTH PRESETS (giống train_image.py)
DEPTH_PRESETS = {
    "normal": {"epochs": 10, "patience": 3, "lr": 1e-5, "batch_size": 32},
    "deep": {"epochs": 20, "patience": 5, "lr": 5e-6, "batch_size": 24},
    "superdeep": {"epochs": 30, "patience": 7, "lr": 1e-6, "batch_size": 16},
}

# ========== GRAD-CAM IMPROVED ==========
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Grad-CAM improved version - tránh lỗi RGBA
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs[0],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    
    if grads is None:
        print("Warning: Gradients are None, returning zero heatmap")
        return np.zeros(IMG_SIZE)
        
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE)
    heatmap = tf.squeeze(heatmap)
    return heatmap.numpy()

def save_and_overlay_heatmap(orig_path, heatmap, out_path, alpha=0.4):
    """
    Lưu heatmap - tránh lỗi RGBA
    """
    orig = Image.open(orig_path).convert("RGB").resize(IMG_SIZE)
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE)
    heatmap_img = heatmap_img.convert("RGBA")
    
    cmap = plt.get_cmap("jet")
    colored = cmap(np.array(heatmap_img)[:, :, 0] / 255.0)
    colored = np.uint8(colored * 255)
    colored_img = Image.fromarray(colored).convert("RGBA")
    
    overlay = Image.blend(orig.convert("RGBA"), colored_img, alpha=alpha)
    overlay = overlay.convert("RGB")  # CHUYỂN VỀ RGB TRƯỚC KHI LƯU
    overlay.save(out_path, "JPEG")

# ========== DATA PREPARATION IMPROVED ==========
def gather_samples_balanced(dataset_root: str, n_samples: int, mode: str = "random"):
    """
    Lấy mẫu cân bằng real/fake như train_image.py
    """
    real_paths = sorted(glob(os.path.join(dataset_root, "real", "*")))
    fake_paths = sorted(glob(os.path.join(dataset_root, "fake", "*")))
    
    print(f"Found {len(real_paths)} real images and {len(fake_paths)} fake images")
    
    # Tính số lượng mỗi class (cân bằng)
    n_per_class = n_samples // 2 if n_samples else None
    
    if n_per_class:
        # Kiểm tra nếu có đủ ảnh cho mỗi class
        available_real = min(len(real_paths), n_per_class)
        available_fake = min(len(fake_paths), n_per_class)
        
        if mode == "random":
            np.random.shuffle(real_paths)
            np.random.shuffle(fake_paths)
        elif mode == "newest":
            real_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            fake_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        real_paths = real_paths[:available_real]
        fake_paths = fake_paths[:available_fake]
        
        print(f"Using {available_real} real images and {available_fake} fake images")
        
        total_available = available_real + available_fake
        if total_available < n_samples:
            print(f"Warning: Only {total_available} images available, less than requested {n_samples}")
    
    samples = [(p, "real") for p in real_paths] + [(p, "fake") for p in fake_paths]
    
    if mode == "random":
        np.random.shuffle(samples)
    
    return samples

def create_dirs(base_out):
    """Tạo thư mục output"""
    os.makedirs(base_out, exist_ok=True)
    subdirs = ["test_heatmaps", "plots"]
    for s in subdirs:
        os.makedirs(os.path.join(base_out, s), exist_ok=True)

# ========== MAIN TRAIN FUNCTION IMPROVED ==========
def continue_train(selected_model_path, num_samples=1000, mode="random", train_ratio=0.8,depth ="normal", seed=42):
    
      # Kiểm tra depth
    assert depth in DEPTH_PRESETS, f"depth must be one of {list(DEPTH_PRESETS.keys())}"
    cfg = DEPTH_PRESETS[depth]
    
    # Lấy config từ depth
    BATCH_SIZE = cfg["batch_size"]
    EPOCHS = cfg["epochs"]
    PATIENCE = cfg["patience"]
    LEARNING_RATE = cfg["lr"]
    
    print(f"[INFO] Using depth preset: {depth}")
    print(f"[INFO] Config: {EPOCHS} epochs, {BATCH_SIZE} batch_size, {LEARNING_RATE} lr")
    # 1️⃣ Load model cũ
    print(f"[INFO] Loading model: {selected_model_path}")
    model = load_model(selected_model_path)
    
    # Cập nhật learning rate cho fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 2️⃣ Tạo folder output mới
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_model_name = os.path.basename(os.path.dirname(selected_model_path))
    output_dir = os.path.join(MODEL_DIR, f"{timestamp}_continue_from_{base_model_name}")
    create_dirs(output_dir)

    # 3️⃣ Lấy dữ liệu cân bằng
    print("[INFO] Gathering balanced samples...")
    samples = gather_samples_balanced(DATASET_DIR, num_samples, mode)
    
    # Thống kê
    real_count = sum(1 for _, label in samples if label == "real")
    fake_count = sum(1 for _, label in samples if label == "fake")
    print(f"Final dataset: {real_count} real, {fake_count} fake (total: {len(samples)})")

    # Lưu samples list
    df_samples = pd.DataFrame(samples, columns=["filepath", "label"])
    sample_list_path = os.path.join(output_dir, "samples_list.xlsx")
    df_samples.to_excel(sample_list_path, index=False)

    # 4️⃣ Chia train/test với stratification
    filepaths = df_samples["filepath"].tolist()
    labels = df_samples["label"].tolist()
    
    fp_train, fp_test, y_train, y_test = train_test_split(
        filepaths, labels, train_size=train_ratio, random_state=seed, stratify=labels
    )

    # Chia train/val (10% của train)
    if len(fp_train) >= 10:
        fp_tr, fp_val, y_tr, y_val = train_test_split(
            fp_train, y_train, test_size=0.1, random_state=seed, stratify=y_train
        )
    else:
        fp_tr, fp_val, y_tr, y_val = fp_train, [], y_train, []

    # Thống kê split
    train_real = sum(1 for label in y_tr if label == "real")
    train_fake = sum(1 for label in y_tr if label == "fake")
    val_real = sum(1 for label in y_val if label == "real")
    val_fake = sum(1 for label in y_val if label == "fake")
    test_real = sum(1 for label in y_test if label == "real")
    test_fake = sum(1 for label in y_test if label == "fake")
    
    print(f"Train set: {train_real} real, {train_fake} fake (total: {len(y_tr)})")
    print(f"Val set: {val_real} real, {val_fake} fake (total: {len(y_val)})")
    print(f"Test set: {test_real} real, {test_fake} fake (total: {len(y_test)})")

    # 5️⃣ Data Generator với augmentation nhẹ
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_gen = train_datagen.flow_from_dataframe(
        pd.DataFrame({"filepath": fp_tr, "label": y_tr}),
        x_col="filepath", y_col="label",
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=True
    )
    
    val_gen = val_test_datagen.flow_from_dataframe(
        pd.DataFrame({"filepath": fp_val, "label": y_val}),
        x_col="filepath", y_col="label", 
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False
    ) if len(fp_val) > 0 else None

    test_gen = val_test_datagen.flow_from_dataframe(
        pd.DataFrame({"filepath": fp_test, "label": y_test}),
        x_col="filepath", y_col="label",
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False
    )

    # 6️⃣ Callbacks
    csv_logger = CSVLogger(os.path.join(output_dir, "training_log.csv"))
    checkpoint = ModelCheckpoint(
        os.path.join(output_dir, "model_checkpoint.h5"),
        save_best_only=True, monitor="val_loss" if val_gen else "loss"
    )
    early_stop = EarlyStopping(
        monitor="val_loss" if val_gen else "loss",
        patience=PATIENCE, restore_best_weights=True
    )

    callbacks = [csv_logger, checkpoint, early_stop]

    # 7️⃣ Training
    print("[INFO] Continue training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # 8️⃣ Lưu model final
    model.save(os.path.join(output_dir, "model_final.h5"))

    # 9️⃣ Evaluation chi tiết
    print("[INFO] Evaluating...")
    test_gen.reset()
    y_pred_proba = model.predict(test_gen)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_gen.classes

    # Classification report
    report = classification_report(y_true, y_pred, target_names=["real", "fake"], output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Metrics đầy đủ
    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision_real": float(report["real"]["precision"]),
        "recall_real": float(report["real"]["recall"]),
        "f1_real": float(report["real"]["f1-score"]),
        "precision_fake": float(report["fake"]["precision"]),
        "recall_fake": float(report["fake"]["recall"]),
        "f1_fake": float(report["fake"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": cm,
        "dataset_stats": {
            "total_samples": len(samples),
            "real_count": real_count,
            "fake_count": fake_count,
            "train_size": len(y_tr),
            "val_size": len(y_val),
            "test_size": len(y_test),
            "train_real": train_real,
            "train_fake": train_fake,
            "val_real": val_real,
            "val_fake": val_fake,
            "test_real": test_real,
            "test_fake": test_fake
        }
    }
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 🔟 Save test results
    test_filenames = [os.path.basename(fp) for fp in fp_test]
    df_test_results = pd.DataFrame({
        "filename": test_filenames,
        "true_label": ["real" if t == 0 else "fake" for t in y_true],
        "pred_label": ["real" if p == 0 else "fake" for p in y_pred],
        "pred_prob": [float(proba[p]) for proba, p in zip(y_pred_proba, y_pred)]
    })
    df_test_results.to_excel(os.path.join(output_dir, "test_results.xlsx"), index=False)

    # 🔟.1 Plots
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    if val_gen:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_acc')
    if val_gen:
        plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "training_curves.png"))
    plt.close()

    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_true, y_pred), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['real', 'fake'],
                yticklabels=['real', 'fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "plots", "confusion_matrix.png"), bbox_inches='tight')
    plt.close()

    # 🔟.2 Grad-CAM heatmaps
    print("[INFO] Generating Grad-CAM heatmaps...")
    
    # Tìm last conv layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
    if last_conv_layer_name is None:
        last_conv_layer_name = "block14_sepconv2_act"
    
    heatmap_dir = os.path.join(output_dir, "test_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    for i, (fp, pred) in enumerate(zip(fp_test, y_pred)):
        if i % 10 == 0:
            print(f"Generating heatmap {i+1}/{len(fp_test)}")
            
        img = tf.keras.preprocessing.image.load_img(fp, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=pred)
        
        out_name = os.path.splitext(os.path.basename(fp))[0] + "_heatmap.jpg"
        out_path = os.path.join(heatmap_dir, out_name)
        save_and_overlay_heatmap(fp, heatmap, out_path)

    print(f"[DONE] Continue training finished. Results saved to: {output_dir}")
    return output_dir

# ========== CLI ==========
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model to continue training")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to use")
    parser.add_argument("--sampling_mode", type=str, default="random", choices=["random", "newest"])
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--depth", type=str, default="normal", 
                       choices=list(DEPTH_PRESETS.keys()), help="Training depth preset")
    
    args = parser.parse_args()
    
    output_dir = continue_train(
        selected_model_path=args.model_path,
        num_samples=args.n_samples,
        mode=args.sampling_mode,
        train_ratio=args.train_ratio,
        depth=args.depth
    )
    
    print(f"✅ Training completed! Results in: {output_dir}")
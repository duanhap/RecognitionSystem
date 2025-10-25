import os
import cv2
import json
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# ==========================
# ⚙️ CONFIG với DEPTH PRESETS
# ==========================
DEPTH_PRESETS = {
    "normal": {"epochs": 10, "patience": 3  , "lr": 1e-4, "batch_size": 32},
    "deep": {"epochs": 20, "patience": 5, "lr": 5e-5, "batch_size": 24},
    "superdeep": {"epochs": 30, "patience": 7, "lr": 1e-5, "batch_size": 16},
}

IMG_SIZE = (299, 299)  # Xception requirement
FRAME_INTERVAL = 10
SEED = 42

# ==========================
# 🎬 TRÍCH FRAME TỪ VIDEO (CẢI TIẾN)
# ==========================
def extract_frames_from_video(video_path, output_dir, label, max_frames_per_video=50):
    """Trích xuất frame từ video với giới hạn số lượng"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    count = 0
    
    if not cap.isOpened():
        return frames
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened() and count < max_frames_per_video:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_INTERVAL == 0:
            frame = cv2.resize(frame, IMG_SIZE)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Tạo tên file unique
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            frame_path = os.path.join(output_dir, f"{label}_{video_name}_frame{count:04d}.jpg")
            #cv2.imwrite(frame_path, frame_rgb)
            Image.fromarray(frame_rgb).save(frame_path)
            frames.append((frame_path, label))
            count += 1
        frame_idx += 1
    
    cap.release()
    return frames

def gather_video_samples(dataset_root, n_samples, mode="random"):
    """
    Lấy mẫu video cân bằng real/fake
    """
    real_videos = sorted(glob(os.path.join(dataset_root, "real", "*")))
    fake_videos = sorted(glob(os.path.join(dataset_root, "fake", "*")))
    
    print(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos")
    
    # Cân bằng số lượng video mỗi class
    n_per_class = n_samples // 2 if n_samples else min(len(real_videos), len(fake_videos))
    
    if mode == "random":
        np.random.shuffle(real_videos)
        np.random.shuffle(fake_videos)
    elif mode == "newest":
        real_videos.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        fake_videos.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    real_videos = real_videos[:n_per_class]
    fake_videos = fake_videos[:n_per_class]
    
    videos = [(p, "real") for p in real_videos] + [(p, "fake") for p in fake_videos]
    
    if mode == "random":
        np.random.shuffle(videos)
    
    return videos

def prepare_dataset(dataset_root, n_samples=500, mode="random", frame_output_dir="extracted_frames"):
    """
    Chuẩn bị dataset từ video với sampling
    """
    os.makedirs(frame_output_dir, exist_ok=True)
    
    # Lấy danh sách video cân bằng
    video_samples = gather_video_samples(dataset_root, n_samples, mode)
    
    all_frames = []
    print("Extracting frames from videos...")
    
    for video_path, label in tqdm(video_samples, desc="Processing videos"):
        if video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
            frames = extract_frames_from_video(video_path, frame_output_dir, label)
            all_frames.extend(frames)
    
    df = pd.DataFrame(all_frames, columns=["filepath", "label"])
    
    # Thống kê
    real_count = len(df[df['label'] == 'real'])
    fake_count = len(df[df['label'] == 'fake'])
    print(f"Extracted {real_count} real frames and {fake_count} fake frames (total: {len(df)})")
    
    return df

# ==========================
# 🧠 GRAD-CAM CHO VIDEO FRAMES
# ==========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Tạo heatmap Grad-CAM"""
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
    """Lưu heatmap overlay"""
    orig = Image.open(orig_path).convert("RGB").resize(IMG_SIZE)
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE)
    heatmap_img = heatmap_img.convert("RGBA")
    
    cmap = plt.get_cmap("jet")
    colored = cmap(np.array(heatmap_img)[:, :, 0] / 255.0)
    colored = np.uint8(colored * 255)
    colored_img = Image.fromarray(colored).convert("RGBA")
    
    overlay = Image.blend(orig.convert("RGBA"), colored_img, alpha=alpha)
    overlay = overlay.convert("RGB")
    overlay.save(out_path, "JPEG")

# ==========================
# 🧠 XÂY DỰNG MODEL
# ==========================
def build_model(lr=1e-4):
    base = Xception(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.4)(x)
    output = Dense(2, activation="softmax")(x)  # 2 classes: real, fake
    model = Model(base.input, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# ==========================
# 🚀 TRAINING PIPELINE HOÀN CHỈNH
# ==========================
def create_dirs(base_out):
    """Tạo thư mục output"""
    os.makedirs(base_out, exist_ok=True)
    subdirs = ["test_heatmaps", "plots"]
    for s in subdirs:
        os.makedirs(os.path.join(base_out, s), exist_ok=True)

def run_video_training_pipeline(
    dataset_root="dataset/video",
    n_samples=500,
    sampling_mode="random", 
    depth="normal",
    out_root="models/video",
    train_ratio=0.8,
    seed=42
):
    # Config từ depth preset
    assert depth in DEPTH_PRESETS, f"depth must be one of {list(DEPTH_PRESETS.keys())}"
    cfg = DEPTH_PRESETS[depth]
    
    print(f"Using depth preset: {depth}")
    print(f"Config: {cfg['epochs']} epochs, {cfg['batch_size']} batch_size, {cfg['lr']} lr")

    # 1. Trích xuất frames từ video
    frame_dir = "extracted_frames"
    df = prepare_dataset(dataset_root, n_samples, sampling_mode, frame_dir)
    
    if len(df) == 0:
        raise RuntimeError("No frames extracted from videos!")
    
    # 2. Tạo output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = os.path.join(out_root, f"{timestamp}_Xception_video")
    create_dirs(model_folder)
    
    # Lưu samples list
    df_samples = pd.DataFrame({
        "filepath": df["filepath"],
        "label": df["label"],
        "video_source": df["filepath"].apply(lambda x: os.path.basename(x).split('_')[1])
    })
    df_samples.to_excel(os.path.join(model_folder, "samples_list.xlsx"), index=False)

    # 3. Chia train/test/val
    filepaths = df["filepath"].tolist()
    labels = df["label"].tolist()
    
    fp_train, fp_test, y_train, y_test = train_test_split(
        filepaths, labels, train_size=train_ratio, random_state=seed, stratify=labels
    )
    
    if len(fp_train) >= 10:
        fp_tr, fp_val, y_tr, y_val = train_test_split(
            fp_train, y_train, test_size=0.1, random_state=seed, stratify=y_train
        )
    else:
        fp_tr, fp_val, y_tr, y_val = fp_train, [], y_train, []

    # Thống kê
    train_real = sum(1 for label in y_tr if label == "real")
    train_fake = sum(1 for label in y_tr if label == "fake")
    test_real = sum(1 for label in y_test if label == "real")
    test_fake = sum(1 for label in y_test if label == "fake")
    
    print(f"Train: {train_real} real, {train_fake} fake frames")
    print(f"Test: {test_real} real, {test_fake} fake frames")

    # 4. Data Generator
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_gen = train_datagen.flow_from_dataframe(
        pd.DataFrame({"filepath": fp_tr, "label": y_tr}),
        x_col="filepath", y_col="label",
        target_size=IMG_SIZE, batch_size=cfg["batch_size"],
        class_mode="categorical", shuffle=True
    )
    
    val_gen = val_test_datagen.flow_from_dataframe(
        pd.DataFrame({"filepath": fp_val, "label": y_val}),
        x_col="filepath", y_col="label",
        target_size=IMG_SIZE, batch_size=cfg["batch_size"],
        class_mode="categorical", shuffle=False
    ) if len(fp_val) > 0 else None

    test_gen = val_test_datagen.flow_from_dataframe(
        pd.DataFrame({"filepath": fp_test, "label": y_test}),
        x_col="filepath", y_col="label",
        target_size=IMG_SIZE, batch_size=cfg["batch_size"],
        class_mode="categorical", shuffle=False
    )

    # 5. Build & Train model
    model = build_model(cfg["lr"])
    
    callbacks = [
        EarlyStopping(monitor="val_loss" if val_gen else "loss", patience=cfg["patience"], restore_best_weights=True),
        ModelCheckpoint(os.path.join(model_folder, "model_checkpoint.h5"), save_best_only=True),
        CSVLogger(os.path.join(model_folder, "training_log.csv"))
    ]

    print("Starting training...")
    history = model.fit(
        train_gen,
        epochs=cfg["epochs"],
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Save final model
    model.save(os.path.join(model_folder, "model_final.h5"))

    # 7. Evaluation
    print("Evaluating...")
    test_gen.reset()
    y_pred_proba = model.predict(test_gen)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_gen.classes

    # Metrics đầy đủ
    report = classification_report(y_true, y_pred, target_names=["real", "fake"], output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

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
            "total_frames": len(df),
            "real_frames": len(df[df['label'] == 'real']),
            "fake_frames": len(df[df['label'] == 'fake']),
            "train_size": len(y_tr),
            "test_size": len(y_test),
            "train_real": train_real,
            "train_fake": train_fake,
            "test_real": test_real,
            "test_fake": test_fake
        }
    }
    
    with open(os.path.join(model_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 8. Save test results
    test_filenames = [os.path.basename(fp) for fp in fp_test]
    df_test_results = pd.DataFrame({
        "filename": test_filenames,
        "true_label": ["real" if t == 0 else "fake" for t in y_true],
        "pred_label": ["real" if p == 0 else "fake" for p in y_pred],
        "pred_prob": [float(proba[p]) for proba, p in zip(y_pred_proba, y_pred)],
        "video_source": [os.path.basename(fp).split('_')[1] for fp in fp_test]
    })
    df_test_results.to_excel(os.path.join(model_folder, "test_results.xlsx"), index=False)

    # 9. Plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    if val_gen:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_acc')
    if val_gen:
        plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, "plots", "training_curves.png"))
    plt.close()

    # 10. Grad-CAM Heatmaps
    print("Generating Grad-CAM heatmaps...")
    last_conv_layer_name = "block14_sepconv2_act"
    
    heatmap_dir = os.path.join(model_folder, "test_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Lấy sample frames để tạo heatmap (tránh quá nhiều)
    sample_indices = np.random.choice(len(fp_test), min(20, len(fp_test)), replace=False)
    
    for idx in sample_indices:
        fp = fp_test[idx]
        pred = y_pred[idx]
        
        img = tf.keras.preprocessing.image.load_img(fp, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=pred)
        
        out_name = os.path.splitext(os.path.basename(fp))[0] + "_heatmap.jpg"
        out_path = os.path.join(heatmap_dir, out_name)
        save_and_overlay_heatmap(fp, heatmap, out_path)

    print(f"✅ Video training completed! Results saved to: {model_folder}")
    return model_folder

# ==========================
# 🎯 CLI INTERFACE
# ==========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Deepfake Detection Model on Videos")
    parser.add_argument("--dataset_root", type=str, default="dataset/video", help="Path to video dataset")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of video samples (balanced real/fake)")
    parser.add_argument("--sampling_mode", type=str, default="random", choices=["random", "newest"])
    parser.add_argument("--depth", type=str, default="normal", choices=list(DEPTH_PRESETS.keys()))
    parser.add_argument("--out_root", type=str, default="models/video", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    result_folder = run_video_training_pipeline(
        dataset_root=args.dataset_root,
        n_samples=args.n_samples,
        sampling_mode=args.sampling_mode,
        depth=args.depth,
        out_root=args.out_root,
        train_ratio=args.train_ratio
    )
    print(f"🎉 Training finished! Results in: {result_folder}")
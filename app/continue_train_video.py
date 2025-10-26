"""
continue_train_video.py

Tiếp tục training từ model video đã có sẵn với video-level pooling evaluation
"""

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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# ==========================
# ⚙️ CONFIG với DEPTH PRESETS
# ==========================
DEPTH_PRESETS = {
    "normal": {"epochs": 10, "patience": 3, "lr": 1e-5, "batch_size": 32},
    "deep": {"epochs": 20, "patience": 5, "lr": 5e-6, "batch_size": 24},
    "superdeep": {"epochs": 30, "patience": 7, "lr": 1e-6, "batch_size": 16},
}

IMG_SIZE = (299, 299)
FRAME_INTERVAL = 10
SEED = 42

# ==========================
# 🎬 TRÍCH FRAME TỪ VIDEO (GIỐNG TRAIN_VIDEO)
# ==========================
def extract_frames_from_video(video_path, output_dir, label, max_frames_per_video=50):
    """Trích xuất frame từ video với giới hạn số lượng"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    count = 0
    
    if not cap.isOpened():
        return frames
    
    while cap.isOpened() and count < max_frames_per_video:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_INTERVAL == 0:
            frame = cv2.resize(frame, IMG_SIZE)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            frame_path = os.path.join(output_dir, f"{label}_{video_name}_frame{count:04d}.jpg")
            Image.fromarray(frame_rgb).save(frame_path)
            frames.append((frame_path, label))
            count += 1
        frame_idx += 1
    
    cap.release()
    return frames

def gather_video_samples(dataset_root, n_samples, mode="random"):
    """Lấy mẫu video cân bằng real/fake"""
    real_videos = sorted(glob(os.path.join(dataset_root, "real", "*")))
    fake_videos = sorted(glob(os.path.join(dataset_root, "fake", "*")))
    
    print(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos")
    
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

def prepare_dataset(dataset_root, n_samples=500, mode="random", frame_output_dir="extracted_frames_continue", force_extract=False):
    """Chuẩn bị dataset từ video với khả năng tái sử dụng frames"""
    os.makedirs(frame_output_dir, exist_ok=True)
    
    # Kiểm tra nếu đã có frames extracted trước đó
    existing_frames = glob(os.path.join(frame_output_dir, "*.jpg"))
    
    if not force_extract and len(existing_frames) > 0:
        print(f"Found {len(existing_frames)} existing frames. Loading from disk...")
        all_frames = []
        for frame_path in existing_frames:
            filename = os.path.basename(frame_path)
            # Extract label from filename (format: label_videoname_frameXXXX.jpg)
            label = filename.split('_')[0]
            if label in ['real', 'fake']:
                all_frames.append((frame_path, label))
        
        if all_frames:
            df = pd.DataFrame(all_frames, columns=["filepath", "label"])
            real_count = len(df[df['label'] == 'real'])
            fake_count = len(df[df['label'] == 'fake'])
            print(f"Loaded {real_count} real frames and {fake_count} fake frames (total: {len(df)})")
            return df
    
    # Nếu không có frames cũ hoặc force_extract=True, trích xuất mới
    print("No existing frames found or force_extract=True. Extracting new frames...")
    video_samples = gather_video_samples(dataset_root, n_samples, mode)
    
    all_frames = []
    print("Extracting frames from videos...")
    
    for video_path, label in tqdm(video_samples, desc="Processing videos"):
        if video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
            frames = extract_frames_from_video(video_path, frame_output_dir, label)
            all_frames.extend(frames)
    
    df = pd.DataFrame(all_frames, columns=["filepath", "label"])
    
    real_count = len(df[df['label'] == 'real'])
    fake_count = len(df[df['label'] == 'fake'])
    print(f"Extracted {real_count} real frames and {fake_count} fake frames (total: {len(df)})")
    
    return df

# ==========================
# 🧠 GRAD-CAM CHO VIDEO FRAMES
# ==========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Tạo heatmap Grad-CAM - đã fix warning"""
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
# 🧠 VIDEO-LEVEL PREDICTION & POOLING
# ==========================
def video_level_predictions(model, video_paths, true_labels, frame_output_dir="temp_frames_video_eval_continue", max_frames_per_video=50):
    """
    Dự đoán ở mức video bằng cách pooling kết quả từ các frames
    """
    os.makedirs(frame_output_dir, exist_ok=True)
    video_predictions = {}
    
    for video_path in tqdm(video_paths, desc="Processing videos for video-level prediction"):
        if video_path not in true_labels:
            continue
            
        # Trích xuất frames từ video
        label = "real" if true_labels[video_path] == 0 else "fake"
        frames = extract_frames_from_video(video_path, frame_output_dir, label, max_frames_per_video)
        frame_paths = [f[0] for f in frames]
        
        if not frame_paths:
            continue
            
        # Dự đoán cho từng frame
        frame_probs = []
        for frame_path in frame_paths:
            img = tf.keras.preprocessing.image.load_img(frame_path, target_size=IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            pred = model.predict(img_array, verbose=0)
            frame_probs.append(pred[0])  # [prob_real, prob_fake]
        
        # Áp dụng various pooling strategies
        frame_probs = np.array(frame_probs)
        fake_probs = frame_probs[:, 1]  # Probability of being fake
        
        # Các phương pháp pooling
        pooling_results = {
            "mean": np.mean(fake_probs),
            "max": np.max(fake_probs),
            "median": np.median(fake_probs),
            "q75": np.quantile(fake_probs, 0.75),  # 75th percentile
            "confidence_weighted": np.average(fake_probs, weights=np.abs(fake_probs - 0.5))
        }
        
        video_predictions[video_path] = {
            "frame_predictions": fake_probs.tolist(),
            "pooling_results": pooling_results,
            "num_frames": len(frame_probs),
            "true_label": true_labels[video_path]
        }
    
    # Dọn dẹp thư mục tạm
    for frame_path in glob(os.path.join(frame_output_dir, "*.jpg")):
        os.remove(frame_path)
    if os.path.exists(frame_output_dir):
        os.rmdir(frame_output_dir)
    
    return video_predictions

def evaluate_video_level(video_predictions):
    """
    Đánh giá ở mức video với various pooling strategies
    """
    if not video_predictions:
        return {}
    
    y_true_video = []
    y_pred_video = {}
    
    # Khởi tạo dict cho các pooling strategies
    for strategy in ["mean", "max", "median", "q75", "confidence_weighted"]:
        y_pred_video[strategy] = []
    
    for video_path, pred_data in video_predictions.items():
        y_true_video.append(pred_data["true_label"])
        
        for strategy in y_pred_video.keys():
            prob_fake = pred_data["pooling_results"][strategy]
            pred_label = 1 if prob_fake > 0.5 else 0  # 1=fake, 0=real
            y_pred_video[strategy].append(pred_label)
    
    # Tính metrics cho từng pooling strategy
    video_metrics = {}
    for strategy, y_pred in y_pred_video.items():
        if len(y_true_video) > 0 and len(y_pred) > 0:
            video_metrics[strategy] = {
                "accuracy": accuracy_score(y_true_video, y_pred),
                "precision": precision_score(y_true_video, y_pred, zero_division=0),
                "recall": recall_score(y_true_video, y_pred, zero_division=0),
                "f1": f1_score(y_true_video, y_pred, zero_division=0),
                "confusion_matrix": confusion_matrix(y_true_video, y_pred).tolist()
            }
    
    return video_metrics

def plot_video_level_predictions(video_predictions, model_folder):
    """Visualize frame predictions within videos"""
    os.makedirs(os.path.join(model_folder, "video_plots"), exist_ok=True)
    
    for i, (video_path, pred_data) in enumerate(list(video_predictions.items())[:8]):  # 8 videos đầu
        if i >= 8:  # Giới hạn số lượng plot
            break
            
        frame_probs = pred_data["frame_predictions"]
        
        plt.figure(figsize=(12, 6))
        plt.plot(frame_probs, 'b-', alpha=0.7, label='Frame Fake Probability')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold')
        plt.axhline(y=np.mean(frame_probs), color='g', linestyle='-', 
                   label=f'Mean Pooling: {np.mean(frame_probs):.3f}')
        
        video_name = os.path.basename(video_path)
        true_label = "fake" if pred_data["true_label"] == 1 else "real"
        plt.title(f'Video: {video_name} (True: {true_label})\n'
                 f'Frames: {len(frame_probs)}, Mean Probability: {np.mean(frame_probs):.3f}')
        plt.xlabel('Frame Index')
        plt.ylabel('Fake Probability')
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Lưu plot
        safe_name = "".join(c for c in video_name if c.isalnum() or c in ('-', '_')).rstrip()
        plt.savefig(os.path.join(model_folder, "video_plots", f"video_predictions_{safe_name}.png"), 
                   bbox_inches='tight', dpi=150)
        plt.close()

# ==========================
# 🚀 CONTINUE TRAINING PIPELINE
# ==========================
def create_dirs(base_out):
    """Tạo thư mục output"""
    os.makedirs(base_out, exist_ok=True)
    subdirs = ["test_heatmaps", "plots", "video_plots"]
    for s in subdirs:
        os.makedirs(os.path.join(base_out, s), exist_ok=True)

def continue_video_training(
    model_path,
    dataset_root="dataset/video",
    n_samples=500,
    sampling_mode="random", 
    depth="normal",
    out_root="models/video",
    train_ratio=0.8,
    seed=42,
    force_extract=False,
    pooling_strategy="mean"
):
    # Config từ depth preset
    assert depth in DEPTH_PRESETS, f"depth must be one of {list(DEPTH_PRESETS.keys())}"
    cfg = DEPTH_PRESETS[depth]
    
    print(f"Using depth preset: {depth}")
    print(f"Config: {cfg['epochs']} epochs, {cfg['batch_size']} batch_size, {cfg['lr']} lr")

    # 1. Load model cũ
    print(f"[INFO] Loading model: {model_path}")
    model = load_model(model_path)
    
    # Cập nhật learning rate cho fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg['lr']),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 2. Trích xuất frames từ video (giữ lại cho lần sau)
    frame_dir = "extracted_frames_continue"
    df = prepare_dataset(dataset_root, n_samples, sampling_mode, frame_dir, force_extract)
    
    if len(df) == 0:
        raise RuntimeError("No frames extracted from videos!")
    
    # 3. Tạo output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_model_name = os.path.basename(os.path.dirname(model_path))
    model_folder = os.path.join(out_root, f"{timestamp}_continue_{depth}_{base_model_name}")
    create_dirs(model_folder)
    
    # Lưu samples list
    df_samples = pd.DataFrame({
        "filepath": df["filepath"],
        "label": df["label"],
        "video_source": df["filepath"].apply(lambda x: os.path.basename(x).split('_')[1])
    })
    df_samples.to_excel(os.path.join(model_folder, "samples_list.xlsx"), index=False)

    # 4. Chia train/test/val
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

    # 5. Data Generator
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

    # 6. Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss" if val_gen else "loss", patience=cfg["patience"], restore_best_weights=True),
        ModelCheckpoint(os.path.join(model_folder, "model_checkpoint.h5"), save_best_only=True),
        CSVLogger(os.path.join(model_folder, "training_log.csv"))
    ]

    # 7. Training tiếp
    print("Continue training...")
    history = model.fit(
        train_gen,
        epochs=cfg["epochs"],
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # 8. Save final model
    model.save(os.path.join(model_folder, "model_final.h5"))

    # 9. Evaluation - FRAME LEVEL
    print("Evaluating at FRAME level...")
    test_gen.reset()
    y_pred_proba = model.predict(test_gen)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_gen.classes

    # Metrics frame-level
    report = classification_report(y_true, y_pred, target_names=["real", "fake"], output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # 9b. Evaluation - VIDEO LEVEL (THÊM MỚI)
    print("Evaluating at VIDEO level...")
    
    # Lấy danh sách video test duy nhất
    test_video_sources = list(set([os.path.basename(fp).split('_')[1] for fp in fp_test]))
    test_video_paths = []
    video_true_labels = {}
    
    # Ánh xạ video source sang đường dẫn và nhãn thật
    for video_source in test_video_sources:
        # Tìm video gốc từ dataset
        for label in ["real", "fake"]:
            video_pattern = os.path.join(dataset_root, label, f"*{video_source}*")
            matching_videos = glob(video_pattern)
            if matching_videos:
                test_video_paths.append(matching_videos[0])
                video_true_labels[matching_videos[0]] = 1 if label == "fake" else 0
                break
    
    # Đánh giá video-level
    video_predictions = video_level_predictions(model, test_video_paths, video_true_labels)
    video_metrics = evaluate_video_level(video_predictions)
    
    # Visualization video-level predictions
    plot_video_level_predictions(video_predictions, model_folder)

    # 10. Lưu kết quả đầy đủ
    metrics = {
        "frame_level": {
            "accuracy": float(np.mean(y_true == y_pred)),
            "precision_real": float(report["real"]["precision"]),
            "recall_real": float(report["real"]["recall"]),
            "f1_real": float(report["real"]["f1-score"]),
            "precision_fake": float(report["fake"]["precision"]),
            "recall_fake": float(report["fake"]["recall"]),
            "f1_fake": float(report["fake"]["f1-score"]),
            "classification_report": report,
            "confusion_matrix": cm,
        },
        "video_level": video_metrics,
        "dataset_stats": {
            "total_frames": len(df),
            "real_frames": len(df[df['label'] == 'real']),
            "fake_frames": len(df[df['label'] == 'fake']),
            "train_size": len(y_tr),
            "test_size": len(y_test),
            "train_real": train_real,
            "train_fake": train_fake,
            "test_real": test_real,
            "test_fake": test_fake,
            "test_videos": len(test_video_paths)
        },
        "training_config": {
            "depth_preset": depth,
            **cfg,
            "pooling_strategy": pooling_strategy,
            "force_extract": force_extract,
            "base_model": model_path
        }
    }
    
    with open(os.path.join(model_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 11. Save test results chi tiết
    test_filenames = [os.path.basename(fp) for fp in fp_test]
    df_test_results = pd.DataFrame({
        "filename": test_filenames,
        "true_label": ["real" if t == 0 else "fake" for t in y_true],
        "pred_label": ["real" if p == 0 else "fake" for p in y_pred],
        "pred_prob_real": [float(proba[0]) for proba in y_pred_proba],
        "pred_prob_fake": [float(proba[1]) for proba in y_pred_proba],
        "video_source": [os.path.basename(fp).split('_')[1] for fp in fp_test]
    })
    df_test_results.to_excel(os.path.join(model_folder, "test_results_frame_level.xlsx"), index=False)

    # Lưu chi tiết video-level results
    video_results = []
    for video_path, pred_data in video_predictions.items():
        video_results.append({
            "video_path": video_path,
            "true_label": "fake" if pred_data["true_label"] == 1 else "real",
            "num_frames": pred_data["num_frames"],
            **{f"prob_fake_{strategy}": pred_data["pooling_results"][strategy] 
               for strategy in pred_data["pooling_results"].keys()},
            "frame_predictions": pred_data["frame_predictions"]
        })
    
    df_video_results = pd.DataFrame(video_results)
    df_video_results.to_excel(os.path.join(model_folder, "test_results_video_level.xlsx"), index=False)

    # 12. Plots
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

    # 13. Grad-CAM Heatmaps
    print("Generating Grad-CAM heatmaps...")
    last_conv_layer_name = "block14_sepconv2_act"
    
    heatmap_dir = os.path.join(model_folder, "test_heatmaps")
    
    # Lấy sample frames để tạo heatmap
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

    print(f"✅ Continue video training completed! Results saved to: {model_folder}")
    
    # In kết quả video-level
    if video_metrics and pooling_strategy in video_metrics:
        best_strategy = video_metrics[pooling_strategy]
        print(f"\n🎯 VIDEO-LEVEL RESULTS (using {pooling_strategy} pooling):")
        print(f"   Accuracy:  {best_strategy['accuracy']:.4f}")
        print(f"   Precision: {best_strategy['precision']:.4f}")
        print(f"   Recall:    {best_strategy['recall']:.4f}")
        print(f"   F1-Score:  {best_strategy['f1']:.4f}")
    
    return model_folder

# ==========================
# 🎯 CLI INTERFACE
# ==========================
def parse_args():
    parser = argparse.ArgumentParser(description="Continue Training Deepfake Detection Model on Videos")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model to continue training")
    parser.add_argument("--dataset_root", type=str, default="dataset/video", help="Path to video dataset")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of video samples (balanced real/fake)")
    parser.add_argument("--sampling_mode", type=str, default="random", choices=["random", "newest"])
    parser.add_argument("--depth", type=str, default="normal", choices=list(DEPTH_PRESETS.keys()))
    parser.add_argument("--out_root", type=str, default="models/video", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--force_extract", action="store_true", help="Force re-extract frames even if they exist")
    parser.add_argument("--pooling_strategy", type=str, default="mean", 
                       choices=["mean", "max", "median", "q75", "confidence_weighted"],
                       help="Pooling strategy for video-level prediction")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    result_folder = continue_video_training(
        model_path=args.model_path,
        dataset_root=args.dataset_root,
        n_samples=args.n_samples,
        sampling_mode=args.sampling_mode,
        depth=args.depth,
        out_root=args.out_root,
        train_ratio=args.train_ratio,
        force_extract=args.force_extract,
        pooling_strategy=args.pooling_strategy
    )
    print(f"🎉 Continue training finished! Results in: {result_folder}")
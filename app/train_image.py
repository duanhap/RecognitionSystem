"""
train_image.py

Pipeline:
- Chọn N sample từ dataset/image/(real|fake) theo random hoặc newest (cân bằng tỉ lệ real/fake)
- Chia train/test (mặc định 80/20)
- Train Xception (pretrained imagenet, fine-tune top)
- Save outputs:
    - models/<timestamp>/model_final.h5
    - training_log.csv (epoch, train_loss, train_acc, val_loss, val_acc)
    - metrics.json (accuracy, precision, recall, f1, confusion matrix)
    - test_results.xlsx (filename, true_label, pred_label, prob)
    - test_heatmaps/<filename>.jpg (Grad-CAM overlay)
    - plots/loss_curve.png, plots/acc_curve.png
    - samples_list.xlsx (liệt kê file paths đã dùng)
"""

import os
import sys
import argparse
import json
import shutil
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Import config
from core.config import settings

# -------------------------
# Config từ settings
# -------------------------
IMG_SIZE = settings.IMAGE_TRAINING_CONFIG["img_size"]
AUTOTUNE = tf.data.AUTOTUNE

def get_image_training_config():
    """Lấy config training cho image từ settings"""
    return settings.IMAGE_TRAINING_CONFIG


# -------------------------
# Utility functions
# -------------------------
def gather_samples_balanced(dataset_root: str, n_samples: int, mode: str = "random"):
    """
    Lấy mẫu cân bằng giữa real và fake
    dataset_root: path to dataset/image with two subfolders 'real' and 'fake'
    mode: 'random' or 'newest'
    Returns: list of (filepath, label)
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
            # newest by file modified time
            real_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            fake_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        real_paths = real_paths[:available_real]
        fake_paths = fake_paths[:available_fake]
        
        print(f"Using {available_real} real images and {available_fake} fake images")
        
        # Nếu một class không đủ ảnh, điều chỉnh class kia để đạt tổng n_samples
        total_available = available_real + available_fake
        if total_available < n_samples:
            print(f"Warning: Only {total_available} images available, less than requested {n_samples}")
    
    samples = [(p, "real") for p in real_paths] + [(p, "fake") for p in fake_paths]
    
    # Final shuffle nếu là random mode
    if mode == "random":
        np.random.shuffle(samples)
    
    return samples

def create_dirs(base_out):
    os.makedirs(base_out, exist_ok=True)
    subdirs = ["test_heatmaps", "plots"]
    for s in subdirs:
        os.makedirs(os.path.join(base_out, s), exist_ok=True)

def load_and_preprocess_image(path):
    # returns float32 image in [0,1]
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def make_dataset(filepaths, labels, batch_size, shuffle=True):
    paths = [str(p) for p in filepaths]
    y_map = {"real": 0, "fake": 1}
    ys = [y_map[l] for l in labels]
    ds = tf.data.Dataset.from_tensor_slices((paths, ys))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))
    def _map(path, label):
        img = load_and_preprocess_image(path)
        return img, tf.one_hot(label, 2)
    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

def build_model(lr):
    base = Xception(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(2, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=out)
    # compile
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# -------------------------
# Grad-CAM for Keras model
# -------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    img_array: preprocessed image tensor shape (1,H,W,3)
    model: keras model
    returns: heatmap (H,W) float32
    """
    # SỬA: Sử dụng inputs=[model.input] thay vì [model.inputs]
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
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE)
    heatmap = tf.squeeze(heatmap)
    return heatmap.numpy()

def save_and_overlay_heatmap(orig_path, heatmap, out_path, alpha=0.4):
    # orig: path to original image
    orig = Image.open(orig_path).convert("RGB").resize(IMG_SIZE)
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE)
    heatmap_img = heatmap_img.convert("RGBA")
    cmap = plt.get_cmap("jet")
    colored = cmap(np.array(heatmap_img)[:, :, 0] / 255.0)
    colored = np.uint8(colored * 255)
    colored_img = Image.fromarray(colored).convert("RGBA")
    overlay = Image.blend(orig.convert("RGBA"), colored_img, alpha=alpha)
    
    # SỬA: Chuyển về RGB trước khi lưu
    overlay = overlay.convert("RGB")
    overlay.save(out_path)


# -------------------------
# Main pipeline
# -------------------------
def run_pipeline(
    dataset_root=None,
    n_samples=None,
    sampling_mode=None,
    depth=None,
    out_root=None,
    train_ratio=None,
    seed=42
):
       # Lấy config mặc định từ settings
    config = get_image_training_config()
    
    # Sử dụng giá trị từ tham số hoặc mặc định từ config
    dataset_root = dataset_root or config["default_dataset_root"]
    n_samples = n_samples or config["default_n_samples"]
    sampling_mode = sampling_mode or config["default_sampling_mode"]
    depth = depth or config["default_depth"]
    train_ratio = train_ratio or config["default_train_ratio"]
    out_root = out_root or str(settings.TRAINING_OUTPUT_DIRS["image"])
    
    # Lấy depth preset từ config
    depth_presets = config["depth_presets"]
    assert depth in depth_presets, f"depth must be one of {list(depth_presets.keys())}"
    cfg = depth_presets[depth]

    print(f"=== Image Training Configuration ===")
    print(f"Dataset: {dataset_root}")
    print(f"Samples: {n_samples} ({sampling_mode} sampling)")
    print(f"Depth: {depth} (epochs: {cfg['epochs']}, batch: {cfg['batch_size']}, lr: {cfg['lr']})")
    print(f"Train ratio: {train_ratio}")
    print(f"Output: {out_root}")
    print("=" * 40)

    # 1) gather samples - CÂN BẰNG REAL/FAKE
    samples = gather_samples_balanced(dataset_root, n_samples, mode=sampling_mode)
    if len(samples) == 0:
        raise RuntimeError("No samples found.")
    
    # Thống kê số lượng
    real_count = sum(1 for _, label in samples if label == "real")
    fake_count = sum(1 for _, label in samples if label == "fake")
    print(f"Final dataset: {real_count} real, {fake_count} fake (total: {len(samples)})")
    
    # 2) prepare outputs - TẠO MODEL_FOLDER TRƯỚC
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = os.path.join(out_root, f"{timestamp}_Xception")
    create_dirs(model_folder)

    # Lưu config sử dụng
    training_config = {
        "dataset_root": dataset_root,
        "n_samples": n_samples,
        "sampling_mode": sampling_mode,
        "depth": depth,
        "train_ratio": train_ratio,
        "seed": seed,
        "depth_config": cfg,
        "timestamp": timestamp
    }
    
    with open(os.path.join(model_folder, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)

    # LƯU SAMPLES_LIST VÀO MODEL_FOLDER
    df_samples = pd.DataFrame(samples, columns=["filepath", "label"])
    sample_list_path = os.path.join(model_folder, "samples_list.xlsx")
    df_samples.to_excel(sample_list_path, index=False)

    # 3) split train/test
    filepaths = df_samples["filepath"].tolist()
    labels = df_samples["label"].tolist()
    fp_train, fp_test, y_train, y_test = train_test_split(
        filepaths, labels, train_size=train_ratio, random_state=seed, stratify=labels
    )

    # Thống kê split
    train_real = sum(1 for label in y_train if label == "real")
    train_fake = sum(1 for label in y_train if label == "fake")
    test_real = sum(1 for label in y_test if label == "real")
    test_fake = sum(1 for label in y_test if label == "fake")
    
    print(f"Train set: {train_real} real, {train_fake} fake (total: {len(y_train)})")
    print(f"Test set: {test_real} real, {test_fake} fake (total: {len(y_test)})")

    # Optionally further split train -> train/val for validation monitoring
    if len(fp_train) >= 10:
        fp_tr, fp_val, y_tr, y_val = train_test_split(fp_train, y_train, test_size=0.1, random_state=seed, stratify=y_train)
    else:
        fp_tr, fp_val, y_tr, y_val = fp_train, [], y_train, []

    # 4) create datasets
    batch_size = cfg["batch_size"]
    train_ds = make_dataset(fp_tr, y_tr, batch_size, shuffle=True)
    val_ds = make_dataset(fp_val, y_val, batch_size, shuffle=False) if len(fp_val)>0 else None
    test_ds = make_dataset(fp_test, y_test, batch_size, shuffle=False)

    # 5) build model
    model = build_model(cfg["lr"])

    # callbacks
    ckpt_path = os.path.join(model_folder, "model_checkpoint.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss" if val_ds is not None else "loss", patience=cfg["patience"], restore_best_weights=True),
        ModelCheckpoint(ckpt_path, save_best_only=True, save_weights_only=False),
    ]
    csv_logger = CSVLogger(os.path.join(model_folder, "training_log.csv"))
    callbacks.append(csv_logger)

    # 6) train
    print("Starting training...")
    history = model.fit(
        train_ds,
        epochs=cfg["epochs"],
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    # save final model
    model_final_path = os.path.join(model_folder, "model_final.h5")
    model.save(model_final_path)

    # save plots
    plt.figure(figsize=(8,5))
    plt.plot(history.history.get("loss", []), label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(model_folder, "plots", "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    if "val_accuracy" in history.history:
        plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(os.path.join(model_folder, "plots", "acc_curve.png"))
    plt.close()

    # 7) Evaluate on test set
    print("Evaluating on test set...")
    y_true = []
    y_pred = []
    y_prob = []
    filenames = []
    
    for fp, lbl in zip(fp_test, y_test):
        img = load_and_preprocess_image(fp)
        img_batch = tf.expand_dims(img, axis=0)
        probs = model.predict(img_batch, verbose=0)
        pred = int(np.argmax(probs, axis=1)[0])
        y_pred.append(pred)
        y_prob.append(float(np.max(probs)))
        y_true.append(0 if lbl == "real" else 1)
        filenames.append(fp)

    # classification report
    report = classification_report(y_true, y_pred, target_names=["real", "fake"], output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
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
            "train_size": len(y_train),
            "test_size": len(y_test),
            "train_real": train_real,
            "train_fake": train_fake,
            "test_real": test_real,
            "test_fake": test_fake
        }
    }
    with open(os.path.join(model_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # save test results xlsx
    df_test = pd.DataFrame({
        "filename": filenames,
        "true_label": ["real" if t==0 else "fake" for t in y_true],
        "pred_label": ["real" if p==0 else "fake" for p in y_pred],
        "pred_prob": y_prob
    })
    df_test.to_excel(os.path.join(model_folder, "test_results.xlsx"), index=False)

    # 8) Grad-CAM heatmaps for each test image
    print("Generating Grad-CAM heatmaps...")
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break
    if last_conv_layer_name is None:
        last_conv_layer_name = "block14_sepconv2" if "block14_sepconv2" in [l.name for l in model.layers] else model.layers[-1].name

    heatmaps_dir = os.path.join(model_folder, "test_heatmaps")
    os.makedirs(heatmaps_dir, exist_ok=True)

    # VÀ TRONG PHẦN GRAD-CAM LOOP:
    for fp, p in zip(filenames, y_pred):
        img = load_and_preprocess_image(fp)
        img_batch = tf.expand_dims(img, axis=0)
        heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer_name, pred_index=p)
        out_name = os.path.splitext(os.path.basename(fp))[0] + "_heatmap.jpg"
        out_path = os.path.join(heatmaps_dir, out_name)
        save_and_overlay_heatmap(fp, heatmap, out_path)

    print("Training pipeline finished. Results saved to:", model_folder)
    return model_folder

# -------------------------
# CLI entrypoint
# -------------------------
def parse_args():
    config = get_image_training_config()
    parser = argparse.ArgumentParser(description="Train Image Deepfake Detection Model")
    parser.add_argument("--dataset_root", type=str, default=config["default_dataset_root"], 
                       help="Path to dataset/image folder")
    parser.add_argument("--n_samples", type=int, default=config["default_n_samples"], 
                       help="Number of samples to use (both classes combined)")
    parser.add_argument("--sampling_mode", type=str, default=config["default_sampling_mode"], 
                       choices=["random", "newest"], help="How to select samples")
    parser.add_argument("--depth", type=str, default=config["default_depth"], 
                       choices=list(config["depth_presets"].keys()), help="Training depth preset")
    parser.add_argument("--out_root", type=str, default=str(settings.TRAINING_OUTPUT_DIRS["image"]), 
                       help="Where to store model output")
    parser.add_argument("--train_ratio", type=float, default=config["default_train_ratio"], 
                       help="Train ratio (rest goes to test)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    folder = run_pipeline(
        dataset_root=args.dataset_root,
        n_samples=args.n_samples,
        sampling_mode=args.sampling_mode,
        depth=args.depth,
        out_root=args.out_root,
        train_ratio=args.train_ratio
    )
    print("Done ->", folder)
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from ..core.config import settings

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block14_sepconv2_act", pred_index=None):
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
        return np.zeros((299, 299))
        
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (299, 299))
    heatmap = tf.squeeze(heatmap)
    return heatmap.numpy()

def save_heatmap_image(original_path, heatmap, output_path, alpha=0.4):
    """Lưu heatmap overlay"""
    orig = Image.open(original_path).convert("RGB").resize((299, 299))
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((299, 299))
    heatmap_img = heatmap_img.convert("RGBA")
    
    cmap = plt.get_cmap("jet")
    colored = cmap(np.array(heatmap_img)[:, :, 0] / 255.0)
    colored = np.uint8(colored * 255)
    colored_img = Image.fromarray(colored).convert("RGBA")
    
    overlay = Image.blend(orig.convert("RGBA"), colored_img, alpha=alpha)
    overlay = overlay.convert("RGB")
    overlay.save(output_path, "JPEG")
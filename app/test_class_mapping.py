# test_class_mapping.py
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Tạo sample data giả
sample_data = {
    'filepath': [
        'fake_sample1.jpg', 'real_sample1.jpg', 
        'fake_sample2.jpg', 'real_sample2.jpg'
    ],
    'label': ['fake', 'real', 'fake', 'real']
}

df = pd.DataFrame(sample_data)

# Tạo data generator GIỐNG NHƯ TRONG CODE CHÍNH
datagen = ImageDataGenerator(rescale=1.0/255)

# Giả lập file paths (không cần file thật)
def create_dummy_images():
    for path in sample_data['filepath']:
        # Tạo file ảnh giả
        img = np.random.rand(299, 299, 3) * 255
        img = img.astype(np.uint8)
        tf.keras.preprocessing.image.save_img(path, img)

create_dummy_images()

# Tạo generator
gen = datagen.flow_from_dataframe(
    df,
    x_col="filepath",
    y_col="label", 
    target_size=(299, 299),
    batch_size=2,
    class_mode="categorical",
    shuffle=True
)

print("=== CLASS MAPPING DEBUG ===")
print("Class indices:", gen.class_indices)
print("Classes:", gen.classes)
print("Number of classes:", gen.num_classes)

# Lấy 1 batch để kiểm tra
x_batch, y_batch = next(gen)
print("\n=== BATCH DEBUG ===")
print("Batch labels shape:", y_batch.shape)
print("First batch labels:")
for i in range(len(y_batch)):
    print(f"  Sample {i}: {y_batch[i]} -> Class: {np.argmax(y_batch[i])}")

print("\nExpected: real=0, fake=1")
print("Actual mapping:", gen.class_indices)

# Dọn dẹp
import os
for path in sample_data['filepath']:
    if os.path.exists(path):
        os.remove(path)
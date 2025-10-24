from PIL import Image

def crop_image(input_path: str, output_path: str, crop_info: dict):
    """Cắt ảnh theo toạ độ crop_info: x, y, x2, y2"""
    with Image.open(input_path) as img:
        left = int(crop_info.get("x", 0))
        top = int(crop_info.get("y", 0))
        right = int(crop_info.get("x2", img.width))
        bottom = int(crop_info.get("y2", img.height))

        # Đảm bảo toạ độ hợp lệ
        left = max(0, left)
        top = max(0, top)
        right = min(img.width, right)
        bottom = min(img.height, bottom)

        cropped = img.crop((left, top, right, bottom))
        cropped.save(output_path)
from moviepy.editor import VideoFileClip


def crop_video(input_path: str, output_path: str, crop_info: dict):
    """Cắt đoạn video và vùng theo crop_info: start, end, x, y, x2, y2"""
    start = crop_info.get("start", 0)
    end = crop_info.get("end", None)
    x1 = int(crop_info.get("x", 0))
    y1 = int(crop_info.get("y", 0))
    x2 = int(crop_info.get("x2", 0))
    y2 = int(crop_info.get("y2", 0))

    with VideoFileClip(input_path) as clip: # type: VideoFileClip
        # Cắt đoạn thời gian
        if end:
            clip = clip.subclip(start, end)
        else:
            clip = clip.subclip(start)

        # Cắt khung hình nếu có toạ độ hợp lệ
        if x2 > x1 and y2 > y1:
            clip = clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)

        clip.write_videofile(output_path, codec="libx264", audio_codec="aac", threads=4, verbose=False, logger=None)

import os
import shutil
import tempfile

def save_temp_file(uploaded_file):
    """Lưu file upload tạm thời và trả về đường dẫn"""
    # Tạo thư mục tạm
    temp_dir = tempfile.mkdtemp(prefix="upload_")

    # Tạo đường dẫn file tạm
    temp_path = os.path.join(temp_dir, uploaded_file.filename)

    # Ghi dữ liệu file xuống
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)

    return temp_path

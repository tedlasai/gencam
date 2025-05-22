import cv2
import os
from glob import glob

# Source and destination directories
src_dir = "/home/tedlasai/genCamera/GOPRO2XResults/OursFullSize"
dst_dir = "/home/tedlasai/genCamera/GOPRO2XResults/Ours"

# Helper function to create destination directory and compute dst_path
def get_dst_path(src_path):
    rel_path = os.path.relpath(src_path, src_dir)
    dst_path = os.path.join(dst_dir, rel_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    return dst_path

# Downsample videos
video_paths = glob(os.path.join(src_dir, "**", "*.mp4"), recursive=True)

for src_path in video_paths:
    print(f"Processing video: {src_path}")
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print(f"Failed to open video {src_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    dst_path = get_dst_path(src_path)
    out = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        downsampled = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        out.write(downsampled)

    cap.release()
    out.release()

# Downsample images
image_paths = glob(os.path.join(src_dir, "**", "*.png"), recursive=True)

for src_path in image_paths:
    print(f"Processing image: {src_path}")
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load image {src_path}")
        continue

    height, width = img.shape[:2]
    new_size = (width // 2, height // 2)
    downsampled = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    dst_path = get_dst_path(src_path)
    cv2.imwrite(dst_path, downsampled)

print("✅ Downsampling complete.")

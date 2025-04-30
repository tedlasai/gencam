import cv2
import os
from glob import glob

# Source and destination directories
src_dir = "/home/tedlasai/genCamera/GoProResults/OursFullSize"
dst_dir = "/home/tedlasai/genCamera/GoProResults/Ours"

# Recursively find all .mp4 files in the source directory
video_paths = glob(os.path.join(src_dir, "**", "*.mp4"), recursive=True)

for src_path in video_paths:
    print(f"Processing {src_path}")

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print(f"Failed to open {src_path}")
        continue

    # Get original video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    # Build destination path and make directories as needed
    rel_path = os.path.relpath(src_path, src_dir)
    dst_path = os.path.join(dst_dir, rel_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    out = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        downsampled = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        out.write(downsampled)

    cap.release()
    out.release()

print("✅ Downsampling complete.")

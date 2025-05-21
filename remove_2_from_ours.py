#!/usr/bin/env python3
import os
import cv2

# Source and destination roots
SRC_DIR = '/home/tedlasai/genCamera/GoProResults/Ours'
DST_DIR = '/home/tedlasai/genCamera/GoProResults/Ours7'

for root, dirs, files in os.walk(SRC_DIR):
    # Compute where in DST_DIR this folder should go
    rel = os.path.relpath(root, SRC_DIR)
    dst_root = os.path.join(DST_DIR, rel)
    os.makedirs(dst_root, exist_ok=True)

    for fname in files:
        if not fname.lower().endswith('.mp4'):
            continue

        src_path = os.path.join(root, fname)
        dst_path = os.path.join(dst_root, fname)

        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            print(f"[!] Failed to open {src_path}")
            continue

        # Fetch properties
        fps          = cap.get(cv2.CAP_PROP_FPS)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Prepare writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))

        # Trim last two frames
        to_write = max(0, total_frames - 2)
        written  = 0
        while written < to_write:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            written += 1

        cap.release()
        out.release()
        print(f"[✓] {src_path} → {dst_path}: wrote {written}/{total_frames} frames")

print("All done. Trimmed videos are in:", DST_DIR)

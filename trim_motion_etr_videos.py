#!/usr/bin/env python3
import os
import cv2
import sys

SRC_ROOT = '/home/tedlasai/genCamera/GoProResults/MotionETR'
DST_ROOT = '/home/tedlasai/genCamera/GoProResults/MotionETR7'

for root, dirs, files in os.walk(SRC_ROOT):
    rel = os.path.relpath(root, SRC_ROOT)
    dst_dir = os.path.join(DST_ROOT, rel)
    os.makedirs(dst_dir, exist_ok=True)

    for fname in files:
        if not fname.lower().endswith('.mp4'):
            continue

        src_path = os.path.join(root, fname)
        dst_path = os.path.join(dst_dir, fname)

        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            print(f"[!] Couldn’t open {src_path}", file=sys.stderr)
            continue

        fps          = cap.get(cv2.CAP_PROP_FPS)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # compute how many frames remain after trimming
        frames_to_write = total_frames - 8
        assert frames_to_write == 7, (
            f"{fname}: after removing 4 front + 4 back, expected 7 frames, got {frames_to_write}"
        )

        # skip first 4 frames
        for _ in range(4):
            ret = cap.grab()
            if not ret:
                raise RuntimeError(f"{fname}: failed to skip frame")

        # set up writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))

        # read & write exactly 7 frames
        written = 0
        while written < 7:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"{fname}: unexpected EOF at frame {written}")
            out.write(frame)
            written += 1

        cap.release()
        out.release()
        print(f"[✓] {src_path} → {dst_path}: wrote {written}/7 frames")

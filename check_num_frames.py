import os
import glob
import cv2
import numpy as np

video_dir = '/home/tedlasai/genCamera/GOPROLargeResults/gt'
mp4_paths = sorted(glob.glob(os.path.join(video_dir, '**', '*.mp4'), recursive=True))
for path in mp4_paths:
    frames = []
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print(f"{os.path.basename(path)}: no frames read")
        continue

    video_array = np.stack(frames, axis=0)
    num_frames, height, width, channels = video_array.shape
    print(f"{os.path.basename(path)} → shape: ({num_frames}, {height}, {width}, {channels})")

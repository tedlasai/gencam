#!/usr/bin/env python3
import os

# root of your FullDataset
DATASET_DIR = '/home/tedlasai/genCamera/FullDataset'

# which extensions count as frames
FRAME_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def count_frames(folder):
    return sum(1 for f in os.listdir(folder)
               if os.path.splitext(f)[1].lower() in FRAME_EXTS)

if __name__ == '__main__':
    # loop each category (GOPRO, sports, etc.)
    for category in sorted(os.listdir(DATASET_DIR)):
        cat_path = os.path.join(DATASET_DIR, category)
        if not os.path.isdir(cat_path):
            continue

        lower_dir = os.path.join(cat_path, 'lower_fps_frames')
        if not os.path.isdir(lower_dir):
            continue

        # scan each sequence under lower_fps_frames
        for seq in sorted(os.listdir(lower_dir)):
            seq_path = os.path.join(lower_dir, seq)
            if not os.path.isdir(seq_path):
                continue

            n = count_frames(seq_path)
            tag = "⚠️" if n < 25 else "✔️"
            print(f"[{tag}] {category}/{seq}: {n} frames")

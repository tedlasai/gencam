import cv2
import os
import numpy as np
from pathlib import Path
from glob import glob

def resize_and_pad(frame, target_size):
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

def add_label(frame, text, position=(30, 50), font_scale=1, thickness=2):
    return cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       (255, 255, 255), thickness, cv2.LINE_AA)

def read_video_frames(path, target_size):
    cap = cv2.VideoCapture(str(path))
    frames = []
    if not cap.isOpened():
        print(f"[Warning] Cannot open video: {path}")
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_and_pad(frame, target_size)
        frames.append(frame)
    cap.release()
    return frames

def create_grid_video(paths_dict, output_path):
    labels = {
        'Ours': 'Ours',
        'MotionETR': 'MotionETR',
        'GT': 'GT',
        'Favaro': 'Favaro'
    }

    cap = cv2.VideoCapture(str(paths_dict['Ours']))
    if not cap.isOpened():
        print(f"[Error] Can't open {paths_dict['Ours']}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    target_size = (width, height)
    videos = {key: read_video_frames(path, target_size) for key, path in paths_dict.items()}

    min_len = min(len(frames) for frames in videos.values())
    if min_len == 0:
        print(f"[Warning] Skipping: one or more videos missing for {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define output video writer
    grid_height = height * 2
    grid_width = width * 2
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (grid_width, grid_height))

    for i in range(min_len):
        frames = {k: add_label(videos[k][i], labels[k]) for k in videos}

        top_row = np.hstack([frames['Ours'], frames['MotionETR']])
        bottom_row = np.hstack([frames['GT'], frames['Favaro']])
        grid = np.vstack([top_row, bottom_row])

        writer.write(grid)

    writer.release()
    print(f"✅ Saved to {output_path}")

def find_relative_video_paths(root_dir):
    return sorted([Path(p).relative_to(root_dir) for p in glob(str(root_dir / '**/*.mp4'), recursive=True)])

def main():
    base_dir = Path("/home/tedlasai/genCamera/GoProResults")
    folders = {
        'Ours': base_dir / 'Ours',
        'MotionETR': base_dir / 'MotionETR',
        'GT': base_dir / 'GT',
        'Favaro': base_dir / 'Favaro'
    }
    output_dir = base_dir / 'Merged2x2'

    rel_paths = find_relative_video_paths(folders['Ours'])

    for rel_path in rel_paths:
        paths_dict = {key: folders[key] / rel_path for key in folders}
        if not all(p.exists() for p in paths_dict.values()):
            print(f"[Skipping] Missing file for {rel_path}")
            continue
        output_path = output_dir / rel_path
        create_grid_video(paths_dict, output_path)

if __name__ == "__main__":
    main()

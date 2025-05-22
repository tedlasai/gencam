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

def read_image_sequence(folder, target_size):
    image_paths = sorted(glob(str(folder / "*.png")) + glob(str(folder / "*.jpg")))
    frames = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = resize_and_pad(img, target_size)
        frames.append(img)
    return frames

def create_grid_video(paths_dict, output_path, fps=24):
    labels = {
        'Ours': 'Ours',
        'MotionETR': 'MotionETR',
        'GT': 'GT',
        'Jin': 'Jin'
    }

    # Sample frame from GT
    sample_imgs = list((paths_dict['GT']).glob("*.png")) + list((paths_dict['GT']).glob("*.jpg"))
    if not sample_imgs:
        print(f"[Error] No images in {paths_dict['GT']}")
        return

    sample_img = cv2.imread(str(sample_imgs[0]))
    if sample_img is None:
        print(f"[Error] Can't read sample image from {paths_dict['GT']}")
        return
    height, width = sample_img.shape[:2]
    target_size = (width, height)

    videos = {key: read_image_sequence(path, target_size) for key, path in paths_dict.items()}

    min_len = min(len(frames) for frames in videos.values())
    if min_len == 0:
        print(f"[Warning] Skipping: one or more folders missing images for {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid_height = height * 2
    grid_width = width * 2
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (grid_width, grid_height))

    for i in range(min_len):
        frames = {k: add_label(videos[k][i], labels[k]) for k in videos}
        top_row = np.hstack([frames['Ours'], frames['MotionETR']]) #MOTIONETR
        bottom_row = np.hstack([frames['GT'], frames['Jin']])
        grid = np.vstack([top_row, bottom_row])
        writer.write(grid)

    writer.release()
    print(f"✅ Saved to {output_path}")

def find_sequence_folders(root_dir):
    return sorted([Path(p) for p in glob(str(root_dir / '**/0000*'), recursive=True)])

def main():
    base_dir = Path("/home/tedlasai/genCamera/GOPROResultsImages")
    folders = {
        'Ours': base_dir / 'Ours',
        'MotionETR': base_dir / 'MotionETR', #MOTIONETR
        'GT': base_dir / 'GT',
        'Jin': base_dir / 'Jin'
    }
    output_dir = 'Merged2x2'

    ours_sequence_folders = find_sequence_folders(folders['Ours'])

    for ours_seq_path in ours_sequence_folders:
        relative_subpath = ours_seq_path.relative_to(folders['Ours'])
        paths_dict = {key: folders[key] / relative_subpath for key in folders}

        missing_keys = [k for k, p in paths_dict.items() if not os.path.isdir(p)]
        if missing_keys:
            print(f"[Skipping] Missing folders for {relative_subpath}: {missing_keys}")
            continue

        output_path = output_dir / relative_subpath.parent / (relative_subpath.name + ".mp4")
        create_grid_video(paths_dict, output_path)

if __name__ == "__main__":
    main()

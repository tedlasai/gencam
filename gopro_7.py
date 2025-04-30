import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def gamma_to_linear(img, gamma=2.2):
    return np.power(img, gamma)

def linear_to_gamma(img, gamma=2.2):
    return np.power(img, 1/gamma)

def process_single_blur(args):
    seq_dir, img_files, start_idx, save_dir_blur, save_dir_sharp, window_size = args
    half_window = window_size // 2
    frame_indices = range(start_idx, start_idx + window_size)
    center_idx = start_idx + half_window

    blur_seq_dir = os.path.join(save_dir_blur, os.path.basename(seq_dir))
    sharp_seq_dir = os.path.join(save_dir_sharp, os.path.basename(seq_dir))
    os.makedirs(blur_seq_dir, exist_ok=True)
    os.makedirs(sharp_seq_dir, exist_ok=True)

    blur_img_path = os.path.join(blur_seq_dir, img_files[center_idx])

    # Skip if already exists
    if os.path.exists(blur_img_path):
        return

    imgs = []
    for i in frame_indices:
        img_path = os.path.join(seq_dir, img_files[i])
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32) / 255.0
        img_linear = gamma_to_linear(img)
        imgs.append(img_linear)

    # Average in linear space
    avg_img_linear = np.mean(imgs, axis=0)
    avg_img_gamma = linear_to_gamma(np.clip(avg_img_linear, 0, 1))

    # Save blurred image
    cv2.imwrite(blur_img_path, (avg_img_gamma * 255).astype(np.uint8))

    # Save sharp images
    for i in frame_indices:
        src_path = os.path.join(seq_dir, img_files[i])
        dst_path = os.path.join(sharp_seq_dir, img_files[i])
        if not os.path.exists(dst_path):
            shutil.copyfile(src_path, dst_path)

def process_sequence(seq_dir, save_dir_blur, save_dir_sharp, window_size=7):
    img_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.png')])
    tasks = []

    num_windows = (len(img_files) - window_size) // window_size + 1

    for w in range(num_windows):
        start_idx = w * window_size
        if start_idx + window_size <= len(img_files):
            tasks.append((seq_dir, img_files, start_idx, save_dir_blur, save_dir_sharp, window_size))

    if tasks:
        with Pool(cpu_count()) as pool:
            list(tqdm(pool.imap_unordered(process_single_blur, tasks), total=len(tasks), desc=os.path.basename(seq_dir)))

def process_split(split):
    src_root = f"/home/tedlasai/genCamera/GOPRO/{split}"
    save_root_blur = f"/home/tedlasai/genCamera/GOPRO_7/{split}/blur"
    save_root_sharp = f"/home/tedlasai/genCamera/GOPRO_7/{split}/sharp"

    seq_dirs = [os.path.join(src_root, d) for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]

    for seq_dir in seq_dirs:
        process_sequence(seq_dir, save_root_blur, save_root_sharp)

def main():
    for split in ["train", "test"]:
        print(f"Processing {split} set...")
        process_split(split)

if __name__ == "__main__":
    main()

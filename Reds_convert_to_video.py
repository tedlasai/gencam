import os
import numpy as np
import imageio.v3 as iio  # for reading images
import videoio  # assuming this is your custom or pip-installed module

def convert_folder_to_video(folder_path, output_path, fps=120):
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".png")
    ])

    if not image_files:
        return

    frames = []
    for file in image_files:
        img = iio.imread(os.path.join(folder_path, file))  # shape: (H, W, C), dtype: uint8
        frames.append(img)

    video_array = np.stack(frames, axis=0)  # shape: (T, H, W, C)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    videoio.videosave(output_path, video_array, fps=fps)
    print(f"Saved video: {output_path}")

def process_reds_dataset():
    base_input = "REDS_dataset"
    base_output = "REDS_Video"
    subdirs = {
        "train/train_sharp": "train",
        "val/val_sharp": "val"
    }

    for subdir, tag in subdirs.items():
        input_dir = os.path.join(base_input, subdir)
        output_dir = os.path.join(base_output, tag)

        for folder in sorted(os.listdir(input_dir)):
            folder_path = os.path.join(input_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            save_path = os.path.join(output_dir, f"{folder}.mp4")
            convert_folder_to_video(folder_path, save_path)

if __name__ == "__main__":
    process_reds_dataset()

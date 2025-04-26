import cv2
import numpy as np
import os
import imageio.v3 as iio  # for reading images


def cv_videosave(output_path, video_array, fps=30):
    """
    Save a video from a (T, H, W, C) numpy array using OpenCV VideoWriter.
    
    Args:
        output_path (str): Path to save the output video (should end with .mp4 or .avi).
        video_array (np.ndarray): Array of shape (T, H, W, C) and dtype uint8.
        fps (int): Frames per second.
    """
    assert video_array.ndim == 4 and video_array.shape[-1] == 3, "Expected (T, H, W, C=3) array"
    assert video_array.dtype == np.uint8, "Expected dtype=uint8 for video_array"

    T, H, W, C = video_array.shape
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    if not out.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    #flip channel order
    video_array = video_array[:, :, :, ::-1]  # Convert from RGB to BGR cause OpenCV uses BGR format
    for frame in video_array:
        out.write(frame)

    out.release()
    print(f"✅ Saved video: {output_path}")

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
    cv_videosave(output_path, video_array, fps=fps)
    print(f"Saved video: {output_path}")

def process_reds_dataset():
    base_input = "GOPRO"
    base_output = "GOPRO_Video"
    subdirs = {
        #"train": "train",
        "test": "test"
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

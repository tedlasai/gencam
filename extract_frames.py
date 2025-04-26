import cv2
import os
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor

def extract_frames(video_path, output_root):
    video_name = Path(video_path).stem
    frames_dir = os.path.join(output_root, video_name)
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(frames_dir, f"frame_{frame_idx:05d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames from {video_name}")

def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    video_extensions = {'.mp4', '.mov', '.m4v', '.avi'}

    video_files = [input_dir / f for f in os.listdir(input_dir) if f.lower().endswith(tuple(video_extensions))]

    print(f"Found {len(video_files)} videos to process.")

    # Use multiprocessing to parallelize
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(extract_frames, video, output_dir) for video in video_files]

        # Wait for all futures to complete
        for future in futures:
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos into separate folders.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input videos.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted frames.")
    args = parser.parse_args()
    main(args)

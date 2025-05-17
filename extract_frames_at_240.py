import cv2
import os
import numpy as np
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_video(video_path, output_root):
    print(f"Processing video: {video_path}")
    video_name = Path(video_path).stem
    frames_dir = os.path.join(output_root, video_name)
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    frames = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_idx += 1

        
        if len(frames) == 8:
            # stack into shape (8, H, W, C), normalize to [0,1]
            arr = np.stack(frames).astype(np.float32) / 255.0

            # 1) linearize with γ≈2.2
            lin = arr ** 2.2

            # 2) average in linear space
            avg_lin = lin.mean(axis=0)

            # 3) re‐encode with γ≈1/2.2 and back to uint8
            summed_frame = np.clip((avg_lin ** (1/2.2)) * 255.0, 0, 255).astype(np.uint8)

            # write out
            frame_filename = os.path.join(frames_dir, f"frame_{saved_idx:05d}.png")
            cv2.imwrite(frame_filename, summed_frame)
            saved_idx += 1
            frames = []

    # Add the last frame as it is
    if frames:
        last_frame = frames[-1]
        frame_filename = os.path.join(frames_dir, f"frame_{saved_idx:05d}.png")
        cv2.imwrite(frame_filename, last_frame)

    cap.release()
    print(f"Finished processing {video_name}. Saved {saved_idx + 1} frames.")

def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    max_workers = args.max_workers

    os.makedirs(output_dir, exist_ok=True)

    video_extensions = {'.mp4', '.mov', '.m4v', '.avi'}
    video_files = [input_dir / f for f in os.listdir(input_dir) if f.lower().endswith(tuple(video_extensions))]

    print(f"Found {len(video_files)} videos to process.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_video, video, output_dir): video for video in video_files}

        for future in as_completed(futures):
            video = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Video {video} generated an exception: {exc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and process frames from videos.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input videos.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed frames.")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of videos to process in parallel.")
    args = parser.parse_args()
    main(args)

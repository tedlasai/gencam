import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import videoio
import re

def match_triplets(blurry_root, deblurred_root, gt_root):
    triplets = []
    for img_path in blurry_root.rglob('*.png'):
        rel_path = img_path.relative_to(blurry_root).with_suffix('')
        deblurred_video = deblurred_root / rel_path.with_suffix('.mp4')
        gt_video = gt_root / rel_path.with_suffix('.mp4')
        if deblurred_video.exists():
            triplets.append((img_path, deblurred_video, gt_video if gt_video.exists() else None))
    return triplets

def add_label(image, text, position, font_scale=1, thickness=2):
    return cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                       (255, 255, 255), thickness, cv2.LINE_AA)

def resize_and_pad(img, shape):
    return cv2.resize(img, (shape[1], shape[0]))

def create_merged_video(img_path, deblur_path, gt_path, output_path):
    blurry_img = cv2.imread(str(img_path))

    cap_deblur = cv2.VideoCapture(str(deblur_path))
    cap_gt = cv2.VideoCapture(str(gt_path)) if gt_path is not None else None

    fps = cap_deblur.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap_deblur.get(cv2.CAP_PROP_FRAME_COUNT))

    height = int(cap_deblur.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap_deblur.get(cv2.CAP_PROP_FRAME_WIDTH))

    blurry_img = resize_and_pad(blurry_img, (height, width))


    # Extract exposure from filename (e.g., _w03 -> 3/240s)
    match = re.search(r'_w(\d+)', img_path.name)
    if match:
        exposure_fraction = f"{int(match.group(1))}/240s"
        label_text = f"Blurry ({exposure_fraction})"
    else:
        label_text = "Blurry"

    blurry_img = add_label(blurry_img, label_text, (30, 50))

    empty_top_right = np.zeros_like(blurry_img)
    empty_bottom_right = np.zeros((height, width, 3), dtype=np.uint8)

    frames = []

    for i in range(frame_count):
        ret_deblur, frame_deblur = cap_deblur.read()
        frame_gt = None
        if cap_gt:
            ret_gt, frame_gt = cap_gt.read()
        else:
            ret_gt = False

        if not ret_deblur:
            print(f"[Warning] Skipping frame {i} due to deblur read failure.")
            break

        frame_deblur = resize_and_pad(frame_deblur, (height, width))
        frame_deblur = add_label(frame_deblur, "Deblurred", (30, 50))

        if ret_gt and frame_gt is not None:
            frame_gt = resize_and_pad(frame_gt, (height, width))
            frame_gt = add_label(frame_gt, "GT", (30, 50))
        else:
            frame_gt = empty_bottom_right.copy()

        top_row = np.hstack([blurry_img, empty_top_right])
        bottom_row = np.hstack([frame_deblur, frame_gt])
        stacked = np.vstack([top_row, bottom_row])

        frames.append(stacked)

    cap_deblur.release()
    if cap_gt:
        cap_gt.release()

    # (T, H, W, C)
    video_array = np.stack(frames, axis=0)
    videoio.videosave(str(output_path), video_array[:,:,:,::-1], fps=10)

def main(root_path):
    root = Path(root_path)
    blurry_root = root / 'blurry'
    deblurred_root = root / 'deblurred'
    gt_root = root / 'gt'
    output_root = root / 'merged'

    triplets = match_triplets(blurry_root, deblurred_root, gt_root)
    print(f'Found {len(triplets)} triplets.')

    for img_path, deblurred_path, gt_path in triplets:
        rel_path = img_path.relative_to(blurry_root).with_suffix('.mp4')
        output_path = output_root / rel_path
        print(f'Processing: {rel_path}')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        create_merged_video(img_path, deblurred_path, gt_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Path to root directory (contains blurry/, deblurred/, gt/)')
    args = parser.parse_args()
    main(args.root)

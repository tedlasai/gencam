import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from videoio import videoread
import cv2

def compute_psnr(frame1, frame2):
    mse = np.mean((frame1.astype(np.float32) - frame2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# Paths
deblurred_root = Path("/home/tedlasai/genCamera/CroppedOutput/deblurred")
gt_root = Path("/home/tedlasai/genCamera/CroppedOutput/gt")
animation_root = Path("/home/tedlasai/genCamera/CroppedOutput/animation-from-blur")

# Collect all deblurred mp4s
deblurred_files = list(deblurred_root.rglob("*.mp4"))
total_psnr = 0.0
total_anim_psnr = 0.0
video_count = 0
anim_video_count = 0
frame_count = 0
anim_frame_count = 0

for deblur_file in tqdm(deblurred_files):
    rel_path = deblur_file.relative_to(deblurred_root)
    gt_file = gt_root / rel_path
    animation_file = animation_root / rel_path

    if not gt_file.exists():
        print(f"Missing GT file for: {deblur_file}")
        continue

    # Read videos
    deblur_frames = videoread(str(deblur_file))
    gt_frames = videoread(str(gt_file))

    if deblur_frames.shape != gt_frames.shape:
        print(f"Shape mismatch in {deblur_file.name}: {deblur_frames.shape} vs {gt_frames.shape}")
        continue

    video_psnr = 0.0
    for i in range(deblur_frames.shape[0]):
        psnr = compute_psnr(deblur_frames[i], gt_frames[i])
        video_psnr += psnr
    video_psnr /= deblur_frames.shape[0]

    reverse_psnr = 0.0
    for i in range(deblur_frames.shape[0]):
        psnr = compute_psnr(deblur_frames[-i-1], gt_frames[i])
        reverse_psnr += psnr
    reverse_psnr /= deblur_frames.shape[0]

    avg_psnr = max(video_psnr, reverse_psnr)
    print(f"{deblur_file.name} → Deblurred PSNR: {avg_psnr:.2f}")

    total_psnr += avg_psnr
    frame_count += deblur_frames.shape[0]
    video_count += 1

    # Optional comparison with animation-from-blur (if exists)
    if animation_file.exists():
        animation_frames = videoread(str(animation_file))
        if animation_frames.shape != gt_frames.shape:
            print(f"(Anim) Shape mismatch in {animation_file.name}: {animation_frames.shape} vs {gt_frames.shape}")
            continue

        anim_psnr = 0.0
        for i in range(animation_frames.shape[0]):
            psnr = compute_psnr(animation_frames[i], gt_frames[i])
            anim_psnr += psnr
        anim_psnr /= animation_frames.shape[0]

        print(f"{animation_file.name} → Animation PSNR: {anim_psnr:.2f}")
        total_anim_psnr += anim_psnr
        anim_frame_count += animation_frames.shape[0]
        anim_video_count += 1

    print("Total PSNR:", total_psnr)
    print("Total Animation PSNR:", total_anim_psnr)

# Final summary
if frame_count > 0:
    print(f"\n📊 Average Deblurred PSNR over {frame_count} frames ({video_count} videos): {total_psnr / video_count:.2f}")
else:
    print("⚠️ No valid deblurred video pairs found.")

if anim_frame_count > 0:
    print(f"🎞️ Average Animation PSNR over {anim_frame_count} frames ({anim_video_count} videos): {total_anim_psnr / anim_video_count:.2f}")
else:
    print("⚠️ No valid animation video pairs found.")


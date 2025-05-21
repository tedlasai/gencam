import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from videoio import videoread
import cv2
import pdb

def compute_psnr_from_mse(mse: float) -> float:
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# Paths
# deblurred_root = Path("/home/tedlasai/genCamera/BaistCroppedLandscape/deblurred")
# gt_root = Path("/home/tedlasai/genCamera/BaistCroppedLandscape/gt")
# animation_root = Path("/home/tedlasai/genCamera/BaistCroppedLandscape/animation-from-blur")

deblurred_root = Path("/home/tedlasai/genCamera/BaistCroppedOutput/deblurred") #finetuned -27.14, animation-from-blur - 27.17, w/o finetuning - 23.52
gt_root        = Path("/home/tedlasai/genCamera/BaistCroppedOutput/gt")
animation_root = Path("/home/tedlasai/genCamera/BaistCroppedOutput/animation-from-blur") # #deblurred-old

# Collect all deblurred mp4s
deblurred_files = list(deblurred_root.rglob("*.mp4"))
total_psnr = 0.0
total_anim_psnr = 0.0
video_count = 0
anim_video_count = 0
frame_count = 0
anim_frame_count = 0

for deblur_file in tqdm(sorted(deblurred_files)):
    rel_path = deblur_file.relative_to(deblurred_root)
    gt_file = gt_root / rel_path
    animation_file = animation_root / rel_path

    if not gt_file.exists() or not animation_file.exists():
        print(f"Missing GT file for: {deblur_file}")
        continue

    # Read videos
    deblur_frames = videoread(str(deblur_file))  # shape [T, H, W, C]
    gt_frames     = videoread(str(gt_file))

    if deblur_frames.shape != gt_frames.shape:
        print(f"Shape mismatch in {deblur_file.name}: {deblur_frames.shape} vs {gt_frames.shape}")
        continue

    # Compute forward and reverse squared errors per pixel
    diff_fwd = (deblur_frames.astype(np.float32) - gt_frames.astype(np.float32)) ** 2
    diff_rev = (deblur_frames[::-1].astype(np.float32) - gt_frames.astype(np.float32)) ** 2
    diff_fwd = np.mean(diff_fwd, axis=(0,3)) 
    diff_rev = np.mean(diff_rev, axis=(0,3))  


    # Take per-pixel minimum, then average to get MSE
    min_sq_err = np.minimum(diff_fwd, diff_rev)
    mse = min_sq_err.mean()
    avg_psnr = compute_psnr_from_mse(mse)
    
    print(f"{deblur_file.name} → Deblurred Pixel-wise Reverse PSNR: {avg_psnr:.2f}")

    total_psnr += avg_psnr
    frame_count += deblur_frames.shape[0]
    video_count += 1

    # Optional comparison with animation-from-blur (if exists)
    if animation_file.exists():
        animation_frames = videoread(str(animation_file))
        if animation_frames.shape != gt_frames.shape:
            print(f"(Anim) Shape mismatch in {animation_file.name}: {animation_frames.shape} vs {gt_frames.shape}")
            continue

        # Compute pixel-wise reverse errors for animation frames
        diff_fwd_anim = (animation_frames.astype(np.float32) - gt_frames.astype(np.float32)) ** 2
        diff_rev_anim = (animation_frames[::-1].astype(np.float32) - gt_frames.astype(np.float32)) ** 2
        diff_fwd_anim = np.mean(diff_fwd_anim, axis=(0,3)) 
        diff_rev_anim = np.mean(diff_rev_anim, axis=(0,3))  


        # Take per-pixel minimum, then average to get MSE
        min_sq_err_anim = np.minimum(diff_fwd_anim, diff_rev_anim)
        mse_anim = min_sq_err_anim.mean()

        anim_psnr = compute_psnr_from_mse(mse_anim)
        print(f"{animation_file.name} → Animation Pixel-wise Reverse PSNR: {anim_psnr:.2f}")

        total_anim_psnr += anim_psnr
        anim_frame_count += animation_frames.shape[0]
        anim_video_count += 1
    
    print(f"Total PSNR: {total_psnr:.2f} | Total Animation PSNR: {total_anim_psnr:.2f}")

# Final summary
if video_count > 0:
    print(f"\n📊 Average Deblurred PSNR over {frame_count} frames ({video_count} videos): {total_psnr / video_count:.2f}")
else:
    print("⚠️ No valid deblurred video pairs found.")

if anim_video_count > 0:
    print(f"🎞️ Average Animation PSNR over {anim_frame_count} frames ({anim_video_count} videos): {total_anim_psnr / anim_video_count:.2f}")
else:
    print("⚠️ No valid animation video pairs found.")
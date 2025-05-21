import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from videoio import videoread
import cv2

def compute_psnr(frame1, frame2, PIXEL_MAX=255.0):
    mse = np.mean((frame1.astype(np.float32) - frame2.astype(np.float32)) ** 2)
    if mse == 0:
        exit(0)
        return float('inf')
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# # Paths
# deblurred_root = Path("/home/tedlasai/genCamera/BaistCroppedLandscape/deblurred") #22.0 vs 22.75
# gt_root = Path("/home/tedlasai/genCamera/BaistCroppedLandscape/gt")
# animation_root = Path("/home/tedlasai/genCamera/BaistCroppedLandscape/animation-from-blur") # #deblurred-old

# Paths
deblurred_root = Path("/home/tedlasai/genCamera/BaistCroppedOutput/deblurred") #22.0 vs 22.75
gt_root = Path("/home/tedlasai/genCamera/BaistCroppedOutput/gt")
animation_root = Path("/home/tedlasai/genCamera/BaistCroppedOutput/deblurred-old") # #deblurred-old


# Collect all deblurred mp4s
deblurred_files = sorted(list(deblurred_root.rglob("*.mp4")))
total_psnr = 0.0
total_anim_psnr = 0.0
total_blur_psnr = 0.0
total_blur_anim_psnr = 0.0
video_count = 0
anim_video_count = 0
frame_count = 0
anim_frame_count = 0

for deblur_file in tqdm(deblurred_files):
    rel_path = deblur_file.relative_to(deblurred_root)
    gt_file = gt_root / rel_path
    animation_file = animation_root / rel_path

    #print("Processing:", deblur_file)

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
        #print(f"Frame {i}: PSNR = {psnr:.2f}")
    video_psnr /= deblur_frames.shape[0]

    reverse_psnr = 0.0
    for i in range(deblur_frames.shape[0]):
        psnr = compute_psnr(deblur_frames[-i-1], gt_frames[i])
        reverse_psnr += psnr
        #print(f"Reverse Frame {i}: PSNR = {psnr:.2f}")
    reverse_psnr /= deblur_frames.shape[0]

    #sum the frames in gamma space
    deblur_frames = deblur_frames.astype(np.float32) / 255.0
    deblur_frames = np.power(deblur_frames, 2.2)
    deblur_frames_avg = np.mean(deblur_frames, axis=0)
    deblur_frames_avg = np.power(deblur_frames_avg, 1/2.2)

    gt_frames_scaled = gt_frames.astype(np.float32)/255.0
    gt_frames_scaled = np.power(gt_frames_scaled, 2.2)
    gt_frames_avg = np.mean(gt_frames_scaled, axis=0)
    gt_frames_avg = np.power(gt_frames_avg, 1/2.2)

    #Compute PSNR for the average frames
    blurred_psnr = compute_psnr(deblur_frames_avg, gt_frames_avg, PIXEL_MAX=1.0)
    print(f"Avg PSNR: {psnr:.2f}")
    total_blur_psnr += blurred_psnr
    #mean


    #compare to the 

    # import pdb
    # pdb.set_trace()

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

        reverse_psnr = 0.0
        for i in range(deblur_frames.shape[0]):
            psnr = compute_psnr(animation_frames[-i-1], gt_frames[i])
            reverse_psnr += psnr
        reverse_psnr /= deblur_frames.shape[0]

        print(f"Animation PSNR: {anim_psnr:.2f}")
        print(f"Reverse Animation PSNR: {reverse_psnr:.2f}")

        animation_frames = animation_frames.astype(np.float32) / 255.0
        animation_frames = np.power(animation_frames, 2.2)
        animation_frames_avg = np.mean(animation_frames, axis=0)
        animation_frames_avg = np.power(animation_frames_avg, 1/2.2)

        #Compute PSNR for the average frames
        blurred_anim_psnr = compute_psnr(animation_frames_avg, gt_frames_avg, PIXEL_MAX=1.0)
        total_blur_anim_psnr += blurred_anim_psnr

        print(f"Avg PSNR: {psnr:.2f}")
        anim_psnr = max(anim_psnr, reverse_psnr)

        print(f"{animation_file.name} → Animation PSNR: {anim_psnr:.2f}")
        total_anim_psnr += anim_psnr
        anim_frame_count += animation_frames.shape[0]
        anim_video_count += 1

    print("Total PSNR:", total_psnr)
    print("Total Animation PSNR:", total_anim_psnr)

# Final summary
if frame_count > 0:
    print(f"\n📊 Average Deblurred PSNR over {frame_count} frames ({video_count} videos): {total_psnr / video_count:.2f}")
    print(f"Avg Blur PSNR: {total_blur_psnr / video_count:.2f}")
else:
    print("⚠️ No valid deblurred video pairs found.")

if anim_frame_count > 0:
    print(f"🎞️ Average Animation PSNR over {anim_frame_count} frames ({anim_video_count} videos): {total_anim_psnr / anim_video_count:.2f}")
    print(f"Avg Blur Animation PSNR: {total_blur_anim_psnr / anim_video_count:.2f}")
else:
    print("⚠️ No valid animation video pairs found.")


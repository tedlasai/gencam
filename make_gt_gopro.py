import os
import glob
import cv2
import numpy as np
def cv_videosave(output_path, video_array, fps=20,
                 downsample_spatial=1,   # e.g. 2 to halve width & height
                 downsample_temporal=1): # e.g. 2 to keep every 2nd frame
    """
    Save a video from a (T, H, W, C) numpy array using OpenCV VideoWriter,
    with optional spatial and/or temporal downsampling by an integer factor.
    """
    assert video_array.ndim == 4 and video_array.shape[-1] == 3, \
        "Expected (T, H, W, C=3) array"
    assert video_array.dtype == np.uint8, "Expected uint8 array"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    T, H, W, _ = video_array.shape

    # adjust FPS if you’re dropping frames but want the same perceived speed
    out_fps = fps
    if downsample_temporal > 1:
        # if you want to keep video length same, uncomment:
        # out_fps = fps / downsample_temporal
        video_array = video_array[::downsample_temporal]
    
    # prepare writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    # spatially downsample once
    new_size = (W // downsample_spatial, H // downsample_spatial)
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, new_size)

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {output_path}")

    # BGR conversion + per-frame resize
    for frame in video_array:
        # frame: H×W×3, RGB uint8
        bgr = frame[..., ::-1]
        if downsample_spatial > 1:
            bgr = cv2.resize(bgr, new_size, interpolation=cv2.INTER_NEAREST)
        writer.write(bgr)

    writer.release()


# ─── PARAMETERS ───────────────────────────────────────────────────────────────

blur_root  = "/home/tedlasai/genCamera/GOPRO_7/test/blur"
sharp_root = "/home/tedlasai/genCamera/GOPRO_7/test/sharp"
out_root   = "/home/tedlasai/genCamera/GoProResults/GT"
half_win   = 3       # ±3 → 7-frame window
fps        = 20      # output frame rate

# ─── PROCESS ─────────────────────────────────────────────────────────────────

os.makedirs(out_root, exist_ok=True)

for blur_dir, _, _ in os.walk(blur_root):
    rel       = os.path.relpath(blur_dir, blur_root)
    sharp_dir = os.path.join(sharp_root, rel)
    out_dir   = os.path.join(out_root,   rel)

    if not os.path.isdir(sharp_dir):
        continue

    for blur_path in sorted(glob.glob(os.path.join(blur_dir, "*.png"))):
        fname      = os.path.basename(blur_path)
        center_idx = int(os.path.splitext(fname)[0])

        # collect sharp frames as uint8 numpy arrays
        frames = []
        for i in range(center_idx-half_win, center_idx+half_win+1):
            sp = os.path.join(sharp_dir, f"{i:06d}.png")
            img = cv2.imread(sp)
            if img is None:
                print(f"⚠️  Missing {sp}, skipping {center_idx:06d}")
                frames = []
                break
            # cv2.imread gives BGR uint8; convert to RGB
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not frames:
            continue

        video_np   = np.stack(frames, axis=0)  # shape (7, H, W, 3)
        out_mp4    = os.path.join(out_dir, f"{center_idx:06d}.mp4")

        cv_videosave(out_mp4, video_np, fps=fps, downsample_spatial=2)

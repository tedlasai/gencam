import os
import cv2
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from videoio import videosave, videoread

# Base directories
input_root = Path("/home/tedlasai/genCamera/BaistFullDatasetResults")
anno_root = Path("/home/tedlasai/genCamera/Animation-from-Blur-main/dataset/b-aist++")
output_root = Path("/home/tedlasai/genCamera/BaistCroppedOutput")

return 0
# Gather all .png and .mp4 files
all_files = list(input_root.rglob("*"))
relevant_files = [f for f in all_files if f.suffix in {".mp4"}]
for file_path in tqdm(relevant_files):
    file_stem = file_path.stem  # e.g., "00000041"

    try:
        # Get relative path after dataset name (e.g., blurry/deblurred/gt)
        rel_path = file_path.relative_to(input_root)
        dataset_subfolder = rel_path.parts[0]  # blurry, deblurred, or gt
        sequence_path = Path(*rel_path.parts[1:-2])  # e.g., gHO_sBM_c01...
        filename = rel_path.name

        # Construct annotation path
        anno_seq_path = anno_root / sequence_path / "blur_anno"
        anno_file = anno_seq_path / (file_stem + ".pkl")
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        continue

    if not anno_file.exists():
        print(f"Missing annotation for {anno_file}")
        continue

    # Load bounding box
    with open(anno_file, "rb") as f:
        bbox = pickle.load(f)
        if isinstance(bbox, dict):  # just in case
           bbox = bbox.get("bbox", None)[0:4]
        if bbox is None or len(bbox) != 4:
            print(f"Invalid bbox in {anno_file}")
            continue
        x_min, y_min, x_max, y_max = bbox
        print("Bounding box:", bbox)
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)
 

    # Prepare output path
    out_path = output_root / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.suffix == ".png":
        img = cv2.imread(str(file_path))
        if img is None:
            print(f"Failed to read image: {file_path}")
            continue
        #resize to 960x720
        # assume img is your ORIGINAL image
        h0, w0 = img.shape[:2]

        # target crop on the 960×720 plane
        x_min_resized, x_max_resized = x_min, x_max
        y_min_resized, y_max_resized = y_min, y_max

        # compute scale factors from original → 960×720
        scale_x = w0 / 960.0
        scale_y = h0 / 720.0

        # map crop coords back to original
        x1 = int(x_min_resized * scale_x)
        x2 = int(x_max_resized * scale_x)
        y1 = int(y_min_resized * scale_y)
        y2 = int(y_max_resized * scale_y)

        # crop original
        crop = img[y1:y2, x1:x2]

        # one resize to final size
        resized = cv2.resize(crop, (160, 192), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out_path), resized)

    elif file_path.suffix == ".mp4":
        frames = videoread(str(file_path))
        if frames is None or len(frames) == 0:
            print(f"Failed to read video: {file_path}")
            continue
        
        # frames: (N, H0, W0, C)
        N, H0, W0, _ = frames.shape

        # your “960×720” crop coordinates
        # assume x_min, x_max, y_min, y_max are in [0..960), [0..720)
        # compute scale factors
        scale_x = W0 / 960.0
        scale_y = H0 / 720.0

        # map them back to original pixels
        x1 = int(x_min * scale_x)
        x2 = int(x_max * scale_x)
        y1 = int(y_min * scale_y)
        y2 = int(y_max * scale_y)

        # crop once from each original frame, then resize to (160×192)
        cropped = frames[:, y1:y2, x1:x2, :]
        resized_frames = np.stack([
            cv2.resize(f, (160, 192), interpolation=cv2.INTER_LINEAR)
            for f in cropped
        ], axis=0)
        os.makedirs(out_path.parent, exist_ok=True)
        print("Saving video to:", str(out_path))
        videosave(str(out_path), resized_frames)

print("✅ Cropping completed.")

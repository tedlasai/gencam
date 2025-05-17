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
output_root = Path("/home/tedlasai/genCamera/CroppedOutput")

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
        print("Original image shape:", img.shape)
        img = cv2.resize(img, (960, 720), interpolation=cv2.INTER_AREA)
        print("Resized image shape:", img.shape)
        crop = img[y_min:y_max, x_min:x_max]
        resized = cv2.resize(crop, (160, 192), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_path), resized)

    elif file_path.suffix == ".mp4":
        frames = videoread(str(file_path))
        if frames is None or len(frames) == 0:
            print(f"Failed to read video: {file_path}")
            continue
        h_vid, w_vid = frames.shape[1:3]
        frames = np.stack([
                cv2.resize(frame, (960, 720), interpolation=cv2.INTER_AREA)
                for frame in frames
            ], axis=0)
        cropped_frames = frames[:, y_min:y_max, x_min:x_max, :]
        resized_frames = np.stack([
            cv2.resize(frame, (160, 192), interpolation=cv2.INTER_LINEAR)
            for frame in cropped_frames
        ], axis=0)
        os.makedirs(out_path.parent, exist_ok=True)
        videosave(str(out_path), resized_frames)

print("✅ Cropping completed.")

from pathlib import Path
import shutil

# ─── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_ROOT  = Path("/home/tedlasai/genCamera/GOPRO_7/train")          # where your blur/ and sharp/ folders are
OUTPUT_ROOT = Path("/home/tedlasai/genCamera/Animation-from-Blur-main/dataset/gopro_train")   # new directory tree will be created here
HALF_WINDOW = 3                             # frames before/after center
# ──────────────────────────────────────────────────────────────────────────────

BLUR_DIR  = INPUT_ROOT / "blur"
SHARP_DIR = INPUT_ROOT / "sharp"

for blur_file in sorted(BLUR_DIR.glob("*/*.png")):
    scene = blur_file.parent.name
    center_idx = int(blur_file.stem)
    
    # create scene dirs under OUTPUT_ROOT
    scene_root = OUTPUT_ROOT / scene
    blur_out  = scene_root / "blur"
    sharp_out = scene_root / "sharp"
    blur_out.mkdir(parents=True, exist_ok=True)
    sharp_out.mkdir(parents=True, exist_ok=True)
    
    # copy blur (keep name)
    shutil.copy2(blur_file, blur_out / blur_file.name)
    
    # copy & rename 7 sharp frames around center
    for seq, offset in enumerate(range(-HALF_WINDOW, HALF_WINDOW + 1), start=1):
        idx = center_idx + offset
        src = SHARP_DIR / scene / f"{idx:06d}.png"
        if not src.exists():
            continue  # skip if missing
        dst_name = f"{center_idx:06d}_{seq:03d}.png"
        shutil.copy2(src, sharp_out / dst_name)

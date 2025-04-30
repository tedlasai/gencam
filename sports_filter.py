import os
import shutil
import re

# Paths
source_folder = "/home/tedlasai/genCamera/sports/all_higher_fps_videos"
destination_folder = "/home/tedlasai/genCamera/sports/higher_fps_videos"
clip_numbers_rtf = "/home/tedlasai/genCamera/SportsGoodIndices.rtf"  # <-- Update this!

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Read clip numbers from .rtf
with open(clip_numbers_rtf, "r") as f:
    content = f.read()
    # Extract 5-digit numbers
    clip_numbers = re.findall(r'\d{5}', content)

# Fix numbers to 4-digit format
fixed_clip_numbers = [clip_num.lstrip('0').zfill(4) for clip_num in clip_numbers]

# Copy matching video files
for clip_num in fixed_clip_numbers:
    filename = f"clip_{clip_num}.mp4"
    src_path = os.path.join(source_folder, filename)
    dest_path = os.path.join(destination_folder, filename)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dest_path)
        print(f"Copied {filename}")
    else:
        print(f"Warning: {filename} not found!")

print("Done!")

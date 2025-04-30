import os
import shutil
from PIL import Image

# Set your paths
input_root = '/home/tedlasai/genCamera/GOPRO_Large'
output_root = 'Gopro_align_data'

splits = ['train', 'test']  # Process both train and test

for split in splits:
    input_split_path = os.path.join(input_root, split)
    output_split_path = os.path.join(output_root, split)

    os.makedirs(output_split_path, exist_ok=True)

    # Walk through each video folder
    for subdir in os.listdir(input_split_path):
        subdir_path = os.path.join(input_split_path, subdir)
        if not os.path.isdir(subdir_path):
            continue

        blur_gamma_folder = os.path.join(subdir_path, 'blur_gamma')
        sharp_folder = os.path.join(subdir_path, 'sharp')

        if not os.path.exists(blur_gamma_folder) or not os.path.exists(sharp_folder):
            continue  # Skip if blur_gamma/sharp folders don't exist

        for img_name in os.listdir(blur_gamma_folder):
            blur_path = os.path.join(blur_gamma_folder, img_name)
            sharp_path = os.path.join(sharp_folder, img_name)

            if not os.path.exists(sharp_path):
                print(f"Warning: Sharp image missing for {img_name}")
                continue

            # Open images
            blur_img = Image.open(blur_path).convert('RGB')
            sharp_img = Image.open(sharp_path).convert('RGB')

            # Concatenate horizontally
            total_width = blur_img.width + sharp_img.width
            max_height = max(blur_img.height, sharp_img.height)

            new_img = Image.new('RGB', (total_width, max_height))
            new_img.paste(blur_img, (0, 0))
            new_img.paste(sharp_img, (blur_img.width, 0))

            # Save
            base_name = f"{subdir}_{img_name}"
            save_path = os.path.join(output_split_path, base_name)
            new_img.save(save_path)
            print(f"Saved merged image: {save_path}")

print("Finished merging blur_gamma and sharp images!")
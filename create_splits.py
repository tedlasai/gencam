import os
import random

random.seed(42)  # For reproducibility

FULLDATASET_BASE = '/home/tedlasai/genCamera/FullDataset'  # Base path to make relative

def process_dataset(base_path, use_official_split=False, official_split_train=None, official_split_test=None):
    lower_fps_path = os.path.join(base_path, 'lower_fps_frames')

    if use_official_split:
        print(f"Processing {base_path} with official split...")
        train_list = []
        test_list = []

        # Handle train split
        for folder in sorted(os.listdir(official_split_train)):
            full_folder_path = os.path.join(lower_fps_path, folder)
            if os.path.isdir(full_folder_path):
                relative_path = os.path.relpath(full_folder_path, FULLDATASET_BASE)
                train_list.append(relative_path)

        # Handle test split
        for folder in sorted(os.listdir(official_split_test)):
            full_folder_path = os.path.join(lower_fps_path, folder)
            if os.path.isdir(full_folder_path):
                relative_path = os.path.relpath(full_folder_path, FULLDATASET_BASE)
                test_list.append(relative_path)
    else:
        print(f"Processing {base_path} with random 70/30 split...")
        all_folders = [os.path.join(lower_fps_path, f) for f in sorted(os.listdir(lower_fps_path))
                       if os.path.isdir(os.path.join(lower_fps_path, f))]
        random.shuffle(all_folders)
        split_idx = int(0.7 * len(all_folders))
        train_folders = all_folders[:split_idx]
        test_folders = all_folders[split_idx:]

        train_list = [os.path.relpath(f, FULLDATASET_BASE) for f in train_folders]
        test_list = [os.path.relpath(f, FULLDATASET_BASE) for f in test_folders]

    # Save splits next to lower_fps_frames parent
    save_dir = base_path
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'train_list.txt'), 'w') as f:
        for item in train_list:
            f.write(item + '\n')

    with open(os.path.join(save_dir, 'test_list.txt'), 'w') as f:
        for item in test_list:
            f.write(item + '\n')
            print(item)

    print(f"Saved {len(train_list)} train and {len(test_list)} test samples at {save_dir}")

# Dataset paths
gopro_base = os.path.join(FULLDATASET_BASE, 'GOPRO')
adobe240_base = os.path.join(FULLDATASET_BASE, 'Adobe240')
reds_base = os.path.join(FULLDATASET_BASE, 'REDS')
kattolab_base = os.path.join(FULLDATASET_BASE, 'kattolab')
sports_base = os.path.join(FULLDATASET_BASE, 'sports')

# Official split paths for GOPRO
gopro_official_train = '/home/tedlasai/genCamera/Motion-ETR-main/GOPRO_Large/train'
gopro_official_test = '/home/tedlasai/genCamera/Motion-ETR-main/GOPRO_Large/test'

# Process all datasets
process_dataset(gopro_base, use_official_split=True, official_split_train=gopro_official_train, official_split_test=gopro_official_test)
process_dataset(adobe240_base, use_official_split=False)
process_dataset(reds_base, use_official_split=False)
process_dataset(kattolab_base, use_official_split=False)
process_dataset(sports_base, use_official_split=False)

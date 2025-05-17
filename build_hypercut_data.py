import os
import shutil
import json


def make_dataset(original_root, new_root):
    frames_per_seq = 7
    half = frames_per_seq // 2
    metadata = {
        "name": "GOPRO",
        "frame_per_seq": frames_per_seq,
        "data": []
    }

    for partition in ["train", "test"]:
        blur_root = os.path.join(original_root, partition, "blur")
        sharp_root = os.path.join(original_root, partition, "sharp")

        if not os.path.isdir(blur_root) or not os.path.isdir(sharp_root):
            continue

        for seq in sorted(os.listdir(blur_root)):
            seq_blur_dir = os.path.join(blur_root, seq)
            seq_sharp_dir = os.path.join(sharp_root, seq)

            if not os.path.isdir(seq_blur_dir) or not os.path.isdir(seq_sharp_dir):
                continue

            for blur_file in sorted(os.listdir(seq_blur_dir)):
                if not blur_file.lower().endswith('.png'):
                    continue

                frame_num = int(os.path.splitext(blur_file)[0])
                start = frame_num - half
                end = frame_num + half

                # build sample folder
                sample_name = f"{seq}_{blur_file.split('.')[0]}"
                sample_dir = os.path.join(new_root, sample_name)
                blur_dest = os.path.join(sample_dir, "blur")
                sharp_dest = os.path.join(sample_dir, "sharp")
                os.makedirs(blur_dest, exist_ok=True)
                os.makedirs(sharp_dest, exist_ok=True)

                # copy blur
                src_blur = os.path.join(seq_blur_dir, blur_file)
                dst_blur = os.path.join(blur_dest, blur_file)
                shutil.copy2(src_blur, dst_blur)

                # build entry
                entry = {
                    "id": sample_name,
                    "order": "ignore",
                    "partition": partition,
                    "blur_path": f"{sample_name}/blur/{blur_file}"
                }

                # copy sharp frames and record paths
                for i, num in enumerate(range(start, end + 1), start=1):
                    fname = f"{num:06d}.png"
                    src_sharp = os.path.join(seq_sharp_dir, fname)
                    if not os.path.exists(src_sharp):
                        raise FileNotFoundError(f"Missing sharp frame: {src_sharp}")
                    dst_sharp = os.path.join(sharp_dest, fname)
                    shutil.copy2(src_sharp, dst_sharp)
                    entry[f"frame{str(i).zfill(3)}_path"] = f"{sample_name}/sharp/{fname}"

                metadata["data"].append(entry)

    # write metadata file
    os.makedirs(new_root, exist_ok=True)
    with open(os.path.join(new_root, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Reorganize GOPRO dataset into per-sample folders and generate metadata.json")
    parser.add_argument('--original_root', default = "/home/tedlasai/genCamera/GOPRO_7", required=False)
    parser.add_argument('--new_root', default="/home/tedlasai/genCamera/HYPERCUT_GOPRO_7", required=False)
    args = parser.parse_args()

    make_dataset(args.original_root, args.new_root)

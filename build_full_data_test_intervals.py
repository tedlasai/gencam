#!/usr/bin/env python3
import os
import pickle

# root of your FullDataset
DATASET_DIR = '/home/tedlasai/genCamera/FullDataset'

# which extensions count as frames
FRAME_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# every interval spec you requested
INTERVAL_SPECS = [
    # 1x mode, various window sizes
    # {"ds": 2,  "de": 2,  "fps": 240, "mode": "1x"},  # size 4
    # {"ds": 3,  "de": 4,  "fps": 240, "mode": "1x"},  # size 7
    # {"ds": 4,  "de": 4,  "fps": 240, "mode": "1x"},  # size 8
    # {"ds": 6,  "de": 6,  "fps": 240, "mode": "1x"},  # size 12
    {"ds": 8,  "de": 8,  "fps": 240, "mode": "1x"},  # size 16

    # 2x mode
    #{"ds": 4,  "de": 4,  "fps": 240, "mode": "2x"},  # size 8
    {"ds": 8,  "de": 8,  "fps": 120, "mode": "2x"},  # size 16

    # large blur
    #{"ds": 16, "de": 16, "fps": 120, "mode": "lb"},  # size 24
    #{"ds": 24, "de": 24, "fps": 80, "mode": "lb"},  # size 48
]

def count_frames(folder):
    return sum(
        1
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in FRAME_EXTS
    )

if __name__ == '__main__':
    all_intervals = []

    # scan each dataset folder (REDS, GOPRO, etc.)
    for category in sorted(os.listdir(DATASET_DIR)):
        cat_path = os.path.join(DATASET_DIR, category)
        if not os.path.isdir(cat_path):
            continue

        test_list = os.path.join(cat_path, 'test_list.txt')
        if not os.path.isfile(test_list):
            continue

        # load the test-set paths for this dataset
        with open(test_list, 'r') as f:
            test_paths = {line.strip() for line in f if line.strip()}

        lower_dir = os.path.join(cat_path, 'lower_fps_frames')
        if not os.path.isdir(lower_dir):
            print(f"No lower_fps_frames under {category}, skipping.")
            continue

        for seq in sorted(os.listdir(lower_dir)):
            seq_path = os.path.join(lower_dir, seq)
            if not os.path.isdir(seq_path):
                continue

            rel = f"{category}/lower_fps_frames/{seq}"
            n = count_frames(seq_path)
            tag = "⚠️" if n < 25 else "✔️"
            print(f"[{tag}] {rel}: {n} frames")

            if rel not in test_paths:
                continue

            # pick centers: two at 1/3 & 2/3 if >=96, else one at 1/2 if >=48
            if n >= 96:
                centers = [n // 3, (2 * n) // 3]
            elif n >= 48:
                centers = [n // 2]
            else:
                print(f"  • too short (<48) for any window, skipping {rel}")
                continue

            # build every interval spec at each center
            for c in centers:
                for spec in INTERVAL_SPECS:
                    # compute in-interval
                    in_start = c - spec["ds"]
                    in_end   = c + spec["de"]
                    # ensure valid
                    if in_start < 0 or in_end >= n:
                        print(f"    – skipping out-of-bounds [{spec['mode']}, {spec['fps']}fps] @ center {c}")
                        continue

                    # compute out-interval scale factor
                    factor = 2 if spec["mode"] == "2x" else 1
                    out_start = c - spec["ds"] * factor
                    out_end   = c + spec["de"] * factor

                    # ensure out-interval valid (optional)
                    if out_start < 0 or out_end >= n:
                        print(f"    – skipping out-of-bounds out-interval [{spec['mode']}] @ center {c}")
                        continue
                    if (spec["mode"] == "1x" and in_start != out_start):
                        print("invalid")
                        exit()

                    all_intervals.append({
                        "video_name": f"{category}/{seq}",
                        "in_start":   in_start,
                        "in_end":     in_end,
                        "out_start":  out_start,
                        "out_end":    out_end,
                        "center":     c,
                        "fps":        spec["fps"],
                        "mode":       spec["mode"],
                        "window_size": spec["ds"] + spec["de"]
                    })

    # dump one unified pickle at the top of FullDataset
    out_pkl = os.path.join(DATASET_DIR, 'intervals_test.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(all_intervals, f)

    print(f"Saved {len(all_intervals)} total intervals to {out_pkl}")

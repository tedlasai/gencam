#!/usr/bin/env python3
"""
Script to create a gt_subset directory by copying ground truth files corresponding
to deblurred outputs.

Usage:
    python3 create_gt_subset.py

Configuration:
    Modify DEBLURRED_ROOT, GT_ROOT, SUBSET_ROOT variables below to match your paths.
"""

import os
import shutil
import sys

# Root paths - update these if your directory structure changes
deblurred_root = "/home/tedlasai/genCamera/BaistCroppedOutput/deblurred"
gt_root        = "/home/tedlasai/genCamera/BaistCroppedOutput/gt"
subset_root    = "/home/tedlasai/genCamera/BaistCroppedOutput/gt_subset"

def main():
    for root, dirs, files in os.walk(deblurred_root):
        # Compute the relative path under the deblurred directory
        rel_dir = os.path.relpath(root, deblurred_root)
        for fname in files:
            # Paths for deblurred file and its ground truth counterpart
            deblurred_path = os.path.join(root, fname)
            gt_path = os.path.join(gt_root, rel_dir, fname)

            if not os.path.exists(gt_path):
                print(f"Warning: GT file not found: {gt_path}", file=sys.stderr)
                continue

            # Make the corresponding directory in the subset folder
            dest_dir = os.path.join(subset_root, rel_dir)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, fname)

            # Copy the GT file to the subset
            shutil.copy2(gt_path, dest_path)
            print(f"Copied {gt_path} -> {dest_path}")

if __name__ == "__main__":
    main()

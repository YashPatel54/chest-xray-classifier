"""
Pulls the chest-xray-pneumonia dataset from Kaggle into data/chest_xray.

Requires a Kaggle API token at ~/.kaggle/kaggle.json (see README for setup).

Run: python download_data.py
"""
import os
import shutil

import kagglehub

from src import config


def main():
    print("Downloading dataset via kagglehub (cached after first run)...")
    cached_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print(f"Downloaded to cache: {cached_path}")

    # kagglehub caches outside the repo; the dataset's own top-level folder
    # is usually named "chest_xray" -- copy it into ./data/ for local use.
    src_dir = os.path.join(cached_path, "chest_xray")
    if not os.path.isdir(src_dir):
        # some kagglehub versions/dataset mirrors nest it one level differently
        candidates = [
            os.path.join(cached_path, d) for d in os.listdir(cached_path)
            if os.path.isdir(os.path.join(cached_path, d))
        ]
        src_dir = candidates[0] if candidates else cached_path

    dest_dir = config.DATA_DIR
    os.makedirs(os.path.dirname(dest_dir), exist_ok=True)

    if os.path.exists(dest_dir):
        print(f"{dest_dir} already exists, skipping copy.")
    else:
        print(f"Copying {src_dir} -> {dest_dir}")
        shutil.copytree(src_dir, dest_dir)

    for split in ("train", "val", "test"):
        split_dir = os.path.join(dest_dir, split)
        if os.path.isdir(split_dir):
            counts = {
                cls: len(os.listdir(os.path.join(split_dir, cls)))
                for cls in config.CLASS_NAMES
                if os.path.isdir(os.path.join(split_dir, cls))
            }
            print(f"{split}: {counts}")
        else:
            print(f"WARNING: expected split dir not found: {split_dir}")

    print("\nDone. Data is ready at", dest_dir)


if __name__ == "__main__":
    main()

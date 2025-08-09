#!/usr/bin/env python3
"""
Prepare a YOLOv8-compatible dataset structure from flat images/ and labels/ folders.

Input folder layout (existing):
  dataset/
    images/  # *.jpg|*.jpeg|*.png
    labels/  # *.txt in YOLO format, same base filename as image

Output folder layout (created):
  dataset/yolo/
    train/images, train/labels
    val/images,   val/labels
    test/images,  test/labels

Splits default to 80/10/10 with a fixed seed for reproducibility.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_image_label_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            label = labels_dir / (p.stem + ".txt")
            if label.exists():
                pairs.append((p, label))
            else:
                # Skip images without labels
                pass
    return pairs


def split_dataset(pairs: List[Tuple[Path, Path]], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1
    rnd = random.Random(seed)
    pairs_shuffled = pairs.copy()
    rnd.shuffle(pairs_shuffled)
    n = len(pairs_shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = pairs_shuffled[:n_train]
    val = pairs_shuffled[n_train:n_train + n_val]
    test = pairs_shuffled[n_train + n_val:]
    return train, val, test


def copy_pairs(pairs: List[Tuple[Path, Path]], dst_images: Path, dst_labels: Path):
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        shutil.copy2(img, dst_images / img.name)
        shutil.copy2(lbl, dst_labels / lbl.name)


def write_dataset_yaml(root: Path, yaml_path: Path):
    # YOLO dataset YAML using relative paths
    content = (
        "# YOLOv8 dataset YAML for cucumber freshness\n"
        f"path: {root.as_posix()}\n"  # dataset root dir (the yolo/ split folder)
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        "  - wilted_cucumber\n"
        "  - fresh cucumber\n"
    )
    yaml_path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLOv8 dataset splits")
    parser.add_argument("--data-root", default="dataset", help="Path to dataset folder containing images/ and labels/")
    parser.add_argument("--out-subdir", default="yolo", help="Name of subfolder to create under dataset for YOLO splits")
    parser.add_argument("--train", type=float, default=0.8, help="Train ratio (0-1)")
    parser.add_argument("--val", type=float, default=0.1, help="Val ratio (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"

    print(f"Scanning images in: {images_dir}")
    pairs = find_image_label_pairs(images_dir, labels_dir)
    if not pairs:
        raise SystemExit("No image/label pairs found. Ensure labels/*.txt exist matching images.")
    print(f"Found {len(pairs)} pairs")

    train, val, test = split_dataset(pairs, train_ratio=args.train, val_ratio=args.val, seed=args.seed)
    print(f"Split -> train: {len(train)}, val: {len(val)}, test: {len(test)}")

    out_root = data_root / args.out_subdir
    # Copy files
    copy_pairs(train, out_root / "train/images", out_root / "train/labels")
    copy_pairs(val, out_root / "val/images", out_root / "val/labels")
    copy_pairs(test, out_root / "test/images", out_root / "test/labels")

    # Write dataset YAML at dataset/cucumber.yaml (referencing dataset/yolo)
    yaml_path = data_root / "cucumber.yaml"
    write_dataset_yaml(out_root, yaml_path)
    print(f"Wrote dataset YAML: {yaml_path}")
    print("Done.")


if __name__ == "__main__":
    main()

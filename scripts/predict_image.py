#!/usr/bin/env python3
"""
Quick prediction script to run inference on an image or directory using the trained model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/cucumber/yolov8n/weights/best.pt", help="Path to trained weights")
    p.add_argument("--source", required=True, help="Image/video/glob/folder path")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="auto")
    p.add_argument("--save", action="store_true", help="Save visualized predictions")
    return p.parse_args()


def main():
    args = parse_args()
    if not Path(args.weights).exists():
        raise SystemExit(f"Weights not found: {args.weights}")
    model = YOLO(args.weights)
    model.predict(source=args.source, imgsz=args.imgsz, device=args.device, save=args.save, conf=0.25)


if __name__ == "__main__":
    main()

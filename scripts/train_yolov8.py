#!/usr/bin/env python3
"""
Train YOLOv8 on the cucumber dataset using Ultralytics.

Usage:
  python scripts/train_yolov8.py --epochs 50 --imgsz 640 --model yolov8n.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset/cucumber.yaml", help="Path to dataset YAML")
    p.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model weights or YAML (e.g. yolov8n.pt)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=-1, help="Batch size (-1 for auto)")
    p.add_argument("--device", default="auto", help="Device: auto, cpu, 0, 0,1, etc.")
    p.add_argument("--project", default="runs/cucumber", help="Project directory for runs")
    p.add_argument("--name", default="yolov8n", help="Run name")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs)")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"Dataset YAML not found: {data}")

    model = YOLO(args.model)

    results = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        pretrained=True,
        verbose=True,
        cos_lr=True,
        amp=True,
        exist_ok=True,
        plots=True,
    )

    # Evaluate best model on val and test
    best = Path(args.project) / args.name / "weights" / "best.pt"
    if best.exists():
        m = YOLO(str(best))
        print("Validating best.pt on val split...")
        m.val(data=str(data), imgsz=args.imgsz, device=args.device, split="val")
        print("Testing best.pt on test split...")
        m.val(data=str(data), imgsz=args.imgsz, device=args.device, split="test")


if __name__ == "__main__":
    main()

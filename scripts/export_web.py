#!/usr/bin/env python3
"""
Export trained YOLOv8 model for web/demo usage.
- ONNX (for onnxruntime-web or server-side)
- TF.js (via ultralytics export)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/cucumber/yolov8n/weights/best.pt")
    p.add_argument("--formats", nargs="*", default=["onnx", "tfjs"], help="Export formats")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def main():
    args = parse_args()
    w = Path(args.weights)
    if not w.exists():
        raise SystemExit(f"Weights not found: {w}")
    model = YOLO(str(w))
    for fmt in args.formats:
        print(f"Exporting to {fmt}...")
        model.export(format=fmt, imgsz=args.imgsz, opset=13)


if __name__ == "__main__":
    main()

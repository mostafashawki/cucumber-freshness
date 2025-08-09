#!/usr/bin/env python3
"""
Test the trained model with a quick inference on a few validation images.
"""
import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    # Check if we have a trained model
    weights_path = Path("runs/cucumber/yolov8n/weights/best.pt")
    if not weights_path.exists():
        print(f"No trained model found at {weights_path}")
        print("Run training first: python3 scripts/train_yolov8.py")
        return
    
    # Load model and test on a few validation images
    model = YOLO(str(weights_path))
    val_images = Path("dataset/yolo/val/images")
    
    if val_images.exists():
        test_images = list(val_images.glob("*.jpg"))[:3]  # Test first 3 images
        for img in test_images:
            print(f"\nTesting {img.name}:")
            results = model.predict(source=str(img), conf=0.25, verbose=False)
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls]
                        print(f"  Detected: {class_name} (confidence: {conf:.2f})")
                else:
                    print("  No detections")

if __name__ == "__main__":
    main()

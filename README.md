# Cucumber Freshness – YOLOv8 + Web App

Detect fresh vs wilted cucumbers using a YOLOv8 model, then run a camera-enabled web app to analyze images live from desktop or mobile.

Classes:
- 0 → `wilted_cucumber`
- 1 → `fresh_cucumber`

## 1) Setup

Prerequisites:
- Python 3.10+
- pip
- Optional: CUDA-capable GPU for faster training (otherwise CPU works)

Install dependencies:

```bash
# Option A: System/user install
pip install -r requirements.txt

# Option B: Virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Note: If `python3 -m venv` fails on Ubuntu/Debian, install venv first: `sudo apt install python3-venv`.

## 2) Dataset

Place your images and YOLO-format labels here:

```
dataset/
  images/   # *.jpg/*.png
  labels/   # *.txt (YOLO format) matching image basenames
```

Create train/val/test splits and dataset YAML:

```bash
python3 scripts/prepare_yolo_dataset.py --data-root dataset --out-subdir yolo --train 0.8 --val 0.1
```

This generates:
- `dataset/yolo/{train,val,test}/{images,labels}`
- `dataset/cucumber.yaml` pointing to those splits

## 3) Train the model

Start training (GPU if available):

```bash
# Recommended baseline
python3 scripts/train_yolov8.py --data dataset/cucumber.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch -1
```

CPU-only (smaller image size/batch for speed):

```bash
python3 scripts/train_yolov8.py --data dataset/cucumber.yaml --epochs 25 --imgsz 416 --batch 4 --device cpu
```

Outputs:
- Runs directory: `runs/cucumber/yolov8n/`
- Best weights: `runs/cucumber/yolov8n/weights/best.pt`

Quick validation after training:

```bash
python3 scripts/test_model.py
```

Quick image inference:

```bash
python3 scripts/predict_image.py --weights runs/cucumber/yolov8n/weights/best.pt --source dataset/yolo/val/images --save
```

## 4) Run the camera web app

Start the Flask server:

```bash
python3 app.py
```

Open in your browser:
- Desktop: http://127.0.0.1:5000
- Mobile on same Wi‑Fi: use the LAN IP shown in the terminal (e.g., `http://192.168.x.x:5000`)

In the page, click “Start Camera” → “Capture & Analyze” or upload an image. Results show with colored boxes and a freshness label.

API endpoints:
- `GET /` – Web UI
- `POST /predict` – JSON body `{ image: "data:image/jpeg;base64,..." }` → returns detections and top prediction
- `GET /health` – App health and model status

## 5) Export for pure browser inference (optional)

Export to ONNX or TensorFlow.js for front-end inference:

```bash
# ONNX (good for onnxruntime-web)
python3 scripts/export_web.py --weights runs/cucumber/yolov8n/weights/best.pt --formats onnx

# TensorFlow.js
python3 scripts/export_web.py --weights runs/cucumber/yolov8n/weights/best.pt --formats tfjs
```

Then integrate ONNX Runtime Web or TF.js in `templates/index.html` to run fully in the browser.

## Project structure (key files)

```
scripts/
  prepare_yolo_dataset.py  # Build YOLO splits + cucumber.yaml
  train_yolov8.py          # Train pipeline
  predict_image.py         # Batch/one-off predictions
  export_web.py            # Exports for web
  test_model.py            # Quick sanity test on a few images
app.py                     # Flask server exposing / and /predict
templates/index.html       # Camera-enabled web UI
dataset/cucumber.yaml      # YOLO dataset config
runs/cucumber/yolov8n/     # Training outputs (weights, plots, etc.)
```

## Troubleshooting

- venv fails to create: `sudo apt install python3-venv`, then retry.
- No GPU available: pass `--device cpu` when training.
- Torch/CUDA mismatch: install a CUDA-enabled Torch matching your driver from https://pytorch.org/get-started/locally/
- Camera blocked: allow camera permissions in the browser; on mobile, use the LAN URL, not `localhost`.

## License

This project uses Ultralytics YOLOv8. See Ultralytics license for details.

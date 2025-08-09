#!/usr/bin/env python3
"""
Flask web app for cucumber freshness detection with camera support.
"""
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "runs/cucumber/yolov8n/weights/best.pt"
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Model not found at {MODEL_PATH}")
        return False
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get image data from request
        image_data = request.json.get('image')
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Run inference
        results = model.predict(source=image, conf=0.25, verbose=False)
        
        predictions = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    predictions.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        # Determine overall result
        if predictions:
            # Get the prediction with highest confidence
            best_pred = max(predictions, key=lambda x: x['confidence'])
            result = {
                'success': True,
                'prediction': best_pred['class'],
                'confidence': best_pred['confidence'],
                'is_fresh': best_pred['class'] == 'fresh cucumber',
                'all_detections': predictions
            }
        else:
            result = {
                'success': True,
                'prediction': 'No cucumber detected',
                'confidence': 0.0,
                'is_fresh': None,
                'all_detections': []
            }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })

if __name__ == '__main__':
    if load_model():
        print("Starting Flask app...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to load model. Please ensure training is complete.")

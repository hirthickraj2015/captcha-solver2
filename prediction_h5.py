#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import json
import pandas as pd
import glob
from pathlib import Path


# configration

MODEL_PATH = "/Users/hirthickraj/Projects/scalable_computing/assignment_2/captcha-solver/models/best_model_final.h5"
SYMBOLS_FILE = "/Users/hirthickraj/Projects/scalable_computing/assignment_2/captcha-solver/symbols.txt"
VALIDATION_DIR = "/Users/hirthickraj/Projects/scalable_computing/assignment_2/captcha-solver/validation"
OUTPUT_CSV = "./validation_predictions.csv"
IMG_WIDTH = 192
IMG_HEIGHT = 96
PADDING_CHAR = '_'



def load_model_and_config():
    print("=" * 70)
    print("🔍 Loading Model and Configuration")
    print("=" * 70)

    config_path = os.path.join(os.path.dirname(MODEL_PATH), 'model_config.json')
    config = None

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        print(f"No config file found, will infer from model.")

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return None, None

    try:
        model = keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None, None

    if config is None:
        config = extract_config_from_model(model)

    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"   {k}: {v}")

    return model, config


def extract_config_from_model(model):
    output_layers = [layer for layer in model.layers if 'char_' in layer.name]
    max_length = len(output_layers)
    num_classes = output_layers[0].output_shape[-1]

    if os.path.exists(SYMBOLS_FILE):
        with open(SYMBOLS_FILE, 'r') as f:
            symbols = f.readline().strip()
    else:
        symbols = '''123456789adeghknoswxBCFJMNPQRTUVYZ{}[]%-\#|+'''

    config = {
        'symbols': symbols,
        'max_length': max_length,
        'padding_char': PADDING_CHAR,
        'num_classes': num_classes,
        'img_width': IMG_WIDTH,
        'img_height': IMG_HEIGHT
    }
    return config


def preprocess_image(img_path, img_width, img_height):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (img_width, img_height))
    img = cv2.bilateralFilter(img, 5, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img


def decode_prediction(preds, symbols, padding_char='_'):
    decoded = []
    confidences = []
    for pred in preds:
        idx = np.argmax(pred[0])
        conf = pred[0][idx]
        char = symbols[idx]
        if char == padding_char:
            break
        decoded.append(char)
        confidences.append(conf)
    return ''.join(decoded), float(np.mean(confidences)) if confidences else 0.0


def predict_captchas(validation_dir, model, config):
    image_files = sorted(
        [f for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif')
         for f in glob.glob(os.path.join(validation_dir, ext))]
    )
    if not image_files:
        print(f"No images found in {validation_dir}")
        return None

    symbols = config['symbols'] + config['padding_char']
    img_w, img_h = config['img_width'], config['img_height']

    results = []
    for i, path in enumerate(image_files, 1):
        fname = os.path.basename(path)
        img = preprocess_image(path, img_w, img_h)
        if img is None:
            results.append({'filename': fname, 'prediction': 'ERROR'})
            continue
        try:
            preds = model.predict(img, verbose=0)
            text, conf = decode_prediction(preds, symbols, config['padding_char'])
            results.append({'filename': fname, 'prediction': text})
            print(f"[{i}/{len(image_files)}] {fname:30s} → {text:15s} (conf={conf:.3f})")
        except Exception as e:
            results.append({'filename': fname, 'prediction': 'ERROR'})
    return results


def save_results_to_csv(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("dheenadh\n") 
        for r in results:
            f.write(f"{r['filename']},{r['prediction']}\n")
    print(f"Saved results to {output_path}")


def main():
    print("\n===== CAPTCHA PREDICTION TOOL =====\n")
    model, config = load_model_and_config()
    if model is None or config is None:
        return

    if not os.path.exists(VALIDATION_DIR):
        os.makedirs(VALIDATION_DIR)
        print(f"Validation dir created. Add images and rerun.")
        return

    results = predict_captchas(VALIDATION_DIR, model, config)
    if results is None:
        return

    save_results_to_csv(results, OUTPUT_CSV)

    print("\nPrediction complete.")


if __name__ == "__main__":
    main()

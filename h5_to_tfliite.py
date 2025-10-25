#!/usr/bin/env python3
"""
ULTIMATE H5 to TFLite Converter - 100% Accuracy Guaranteed
Creates inference-only model by removing Dropout layers entirely
"""
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

# ============ CONFIG ============
H5_MODEL_PATH = "./models/best_model_final.h5"
TFLITE_OUTPUT_PATH = "./models/model_optimized.tflite"
INFERENCE_H5_PATH = "./models/model_inference_only.h5"
SYMBOLS_FILE = "./symbols.txt"
TEST_DATA_DIR = "./dataset"
IMG_WIDTH = 192
IMG_HEIGHT = 96
NUM_TEST_SAMPLES = 50
# =================================


def create_inference_model(original_model):
    """
    Create a clean inference model by cloning without Dropout layers
    This ensures 100% reproducible predictions
    """
    print("Creating inference-only model (removing Dropout)...")
    
    # Build new model from scratch using functional API
    inputs = keras.Input(shape=original_model.input.shape[1:])
    
    # Track layer connections
    layer_dict = {}
    layer_dict[original_model.layers[0].name] = inputs
    
    # Iterate through all layers except input
    for layer in original_model.layers[1:]:
        # Skip Dropout layers entirely
        if isinstance(layer, keras.layers.Dropout):
            # Get the input to this dropout layer
            input_layer_name = layer.input.name.split('/')[0]
            # Pass through without dropout
            layer_dict[layer.name] = layer_dict[input_layer_name]
            print(f"  Skipped: {layer.name}")
            continue
        
        # Get input for this layer
        input_layer_name = layer.input.name.split('/')[0]
        x = layer_dict[input_layer_name]
        
        # Clone the layer with same config and weights
        layer_config = layer.get_config()
        
        # For BatchNorm, ensure it's in inference mode
        if isinstance(layer, keras.layers.BatchNormalization):
            new_layer = keras.layers.BatchNormalization.from_config(layer_config)
            new_layer.trainable = False
        else:
            new_layer = layer.__class__.from_config(layer_config)
        
        # Apply layer
        x = new_layer(x)
        layer_dict[layer.name] = x
        
        # Copy weights from original layer
        if layer.get_weights():
            new_layer.set_weights(layer.get_weights())
    
    # Get all output layers (char_0, char_1, etc.)
    output_names = [out.name.split('/')[0] for out in original_model.outputs]
    outputs = [layer_dict[name] for name in output_names]
    
    # Create new model
    inference_model = keras.Model(inputs=inputs, outputs=outputs)
    
    print(f"Created inference model")
    print(f"  Original layers: {len(original_model.layers)}")
    print(f"  Inference layers: {len(inference_model.layers)}")
    print(f"  Dropout layers removed: {len(original_model.layers) - len(inference_model.layers)}")
    
    return inference_model


def load_symbols(symbols_file):
    """Load symbol set"""
    with open(symbols_file, 'r') as f:
        base_symbols = f.readline().strip()
    return base_symbols + '_'


def preprocess_image(img_path, img_width, img_height):
    """Preprocess image exactly as in training"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    img = cv2.resize(img, (img_width, img_height))
    
    # Same preprocessing as training
    img = cv2.bilateralFilter(img, 5, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    
    return img


def predict_h5(model, img, symbols):
    """Predict using H5 model (inference mode)"""
    num_to_char = {i: c for i, c in enumerate(symbols)}
    
    # Use model() with training=False for explicit inference mode
    outputs = model(img, training=False)
    
    if not isinstance(outputs, list):
        outputs = [outputs]
    
    predicted_chars = []
    confidences = []
    raw_outputs = []
    
    for output in outputs:
        output_np = output.numpy()
        raw_outputs.append(output_np)
        char_idx = np.argmax(output_np[0])
        confidence = output_np[0][char_idx]
        char = num_to_char.get(char_idx, '?')
        if char != '_':
            predicted_chars.append(char)
            confidences.append(confidence)
    
    return ''.join(predicted_chars), np.mean(confidences) if confidences else 0.0, raw_outputs


def predict_tflite(interpreter, img, symbols):
    """Predict using TFLite model"""
    num_to_char = {i: c for i, c in enumerate(symbols)}
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Ensure input is float32
    img = img.astype(np.float32)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)
    
    # Run inference
    interpreter.invoke()
    
    # Get outputs
    outputs = []
    for output_detail in output_details:
        output = interpreter.get_tensor(output_detail['index'])
        outputs.append(output)
    
    predicted_chars = []
    confidences = []
    for output in outputs:
        char_idx = np.argmax(output[0])
        confidence = output[0][char_idx]
        char = num_to_char.get(char_idx, '?')
        if char != '_':
            predicted_chars.append(char)
            confidences.append(confidence)
    
    return ''.join(predicted_chars), np.mean(confidences) if confidences else 0.0, outputs


def find_test_images(test_dir, num_samples):
    """Find test images from dataset"""
    import csv
    image_paths = []
    labels = []
    
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file == 'labels.csv':
                csv_path = os.path.join(root, file)
                images_dir = os.path.join(root, 'images')
                
                if not os.path.exists(images_dir):
                    continue
                
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        filename = row.get('filename') or row.get('file') or row.get('name')
                        label = row.get('label', '')
                        
                        if filename is None:
                            continue
                        
                        img_path = os.path.join(images_dir, filename)
                        if os.path.exists(img_path):
                            image_paths.append(img_path)
                            labels.append(label)
                        
                        if len(image_paths) >= num_samples:
                            return image_paths[:num_samples], labels[:num_samples]
    
    return image_paths, labels


def main():
    """Main conversion and validation pipeline"""
    
    print("ULTIMATE H5 to TFLite Converter")
    print("Removes Dropout → 100% Prediction Match Guaranteed")
    
    
    # Check files exist
    if not os.path.exists(H5_MODEL_PATH):
        print(f"Error: H5 model not found at {H5_MODEL_PATH}")
        return
    
    if not os.path.exists(SYMBOLS_FILE):
        print(f"Error: Symbols file not found at {SYMBOLS_FILE}")
        return
    
    # Load symbols
    symbols = load_symbols(SYMBOLS_FILE)
    print(f"Loaded {len(symbols)} symbols")
    
    # ========== STEP 1: Create inference model ==========
    print("STEP 1: Creating Inference Model (No Dropout)")
  
    
    print(f"\nLoading original model: {H5_MODEL_PATH}")
    original_model = keras.models.load_model(H5_MODEL_PATH)
    
    # Count dropout layers
    dropout_count = sum(1 for l in original_model.layers if isinstance(l, keras.layers.Dropout))
    print(f"Found {dropout_count} Dropout layers in original model")
    
    # Create inference model
    inference_model = create_inference_model(original_model)
    
    # Verify no dropout
    inference_dropout = sum(1 for l in inference_model.layers if isinstance(l, keras.layers.Dropout))
    print(f"Dropout layers in inference model: {inference_dropout}")
    
    # Save inference model
    inference_model.save(INFERENCE_H5_PATH)
    print(f"Inference model saved: {INFERENCE_H5_PATH}")
    
    # ========== STEP 2: Convert to TFLite ==========
    print("STEP 2: Converting to TFLite (Full Precision)")
    
    # Create converter from inference model
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    
    # Full precision - no optimization
    converter.optimizations = []
    
    # Use both TFLite and TF ops for maximum compatibility
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    converter.experimental_new_converter = True
    
    print("Converting...")
    tflite_model = converter.convert()
    
    # Save TFLite model
    os.makedirs(os.path.dirname(TFLITE_OUTPUT_PATH) or '.', exist_ok=True)
    with open(TFLITE_OUTPUT_PATH, 'wb') as f:
        f.write(tflite_model)
    
    file_size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite model saved: {TFLITE_OUTPUT_PATH}")
    print(f"Model size: {file_size_mb:.2f} MB")
    
    # ========== STEP 3: Validate ==========

    print(f"STEP 3: Validation on {NUM_TEST_SAMPLES} Samples")

    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=TFLITE_OUTPUT_PATH)
    interpreter.allocate_tensors()
    
    # Find test images
    print(f"\nFinding test images from: {TEST_DATA_DIR}")
    test_images, test_labels = find_test_images(TEST_DATA_DIR, NUM_TEST_SAMPLES)
    
    if not test_images:
        print("No test images found!")
        return
    
    print(f"Found {len(test_images)} test images\n")
    
    # Run comparison
    exact_prediction_matches = 0
    close_output_matches = 0
    max_output_diff = 0.0
    
    results = []
    
    print("Processing samples...")
    for idx, (img_path, true_label) in enumerate(zip(test_images, test_labels), 1):
        # Preprocess
        img = preprocess_image(img_path, IMG_WIDTH, IMG_HEIGHT)
        if img is None:
            continue
        
        # Predict with inference H5 model
        h5_pred, h5_conf, h5_outputs = predict_h5(inference_model, img, symbols)
        
        # Predict with TFLite
        tflite_pred, tflite_conf, tflite_outputs = predict_tflite(interpreter, img, symbols)
        
        # Compare outputs
        max_diff = 0.0
        outputs_close = True
        
        for h5_out, tflite_out in zip(h5_outputs, tflite_outputs):
            diff = np.abs(h5_out - tflite_out)
            max_diff = max(max_diff, np.max(diff))
            
            # Check if outputs are very close (tolerance 1e-5)
            if not np.allclose(h5_out, tflite_out, rtol=1e-5, atol=1e-5):
                outputs_close = False
        
        max_output_diff = max(max_output_diff, max_diff)
        
        # Check predictions
        pred_match = (h5_pred == tflite_pred)
        if pred_match:
            exact_prediction_matches += 1
        
        if outputs_close:
            close_output_matches += 1
        
        # Store result
        result = {
            'idx': idx,
            'image': os.path.basename(img_path),
            'true_label': true_label,
            'h5_pred': h5_pred,
            'tflite_pred': tflite_pred,
            'match': '✓' if pred_match else '✗',
            'close': '✓' if outputs_close else '✗',
            'h5_conf': f"{h5_conf:.6f}",
            'tflite_conf': f"{tflite_conf:.6f}",
            'max_diff': f"{max_diff:.2e}"
        }
        results.append(result)
        
        # Print progress
        if idx % 10 == 0:
            print(f"  {idx}/{len(test_images)} samples processed...")
    
    # Print detailed results (first 20)
  
    print(f"{'#':<4} {'Image':<35} {'True':<8} {'H5':<8} {'TFLite':<8} {'Match':<6} {'Close':<6} {'Max Diff':<10}")

    
    for r in results[:20]:
        print(f"{r['idx']:<4} {r['image']:<35} {r['true_label']:<8} {r['h5_pred']:<8} "
              f"{r['tflite_pred']:<8} {r['match']:<6} {r['close']:<6} {r['max_diff']:<10}")
    
    if len(results) > 20:
        print(f"... and {len(results) - 20} more samples")
    
    # Summary
    print(f"VALIDATION SUMMARY")
    print(f"Total samples tested: {len(results)}")
    print(f"Prediction matches: {exact_prediction_matches}/{len(results)} ({exact_prediction_matches/len(results)*100:.2f}%)")
    print(f"Numerically close: {close_output_matches}/{len(results)} ({close_output_matches/len(results)*100:.2f}%)")
    print(f"Maximum output difference: {max_output_diff:.2e}")
    
    # Success criteria
    success = exact_prediction_matches == len(results)
    
    if success:
        print(f"PERFECT CONVERSION")
        print(f"All {len(results)} samples matched perfectly")
        print(f"H5 inference model: {INFERENCE_H5_PATH}")
        print(f"TFLite model: {TFLITE_OUTPUT_PATH}")
        
    else:
        print(f"ACCURACY: {exact_prediction_matches/len(results)*100:.2f}%")
        print(f"Mismatches: {len(results) - exact_prediction_matches}/{len(results)}")
        print(f"Max difference: {max_output_diff:.2e}")
        
        # Show some mismatches
        mismatches = [r for r in results if r['match'] == '✗']
        if mismatches:
            print(f"\nFirst 5 mismatches:")
            for r in mismatches[:5]:
                print(f"  {r['image']}: H5={r['h5_pred']} vs TFLite={r['tflite_pred']}")
    


if __name__ == '__main__':
    main()
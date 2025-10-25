#!/usr/bin/env python3


import os
import numpy as np
import tensorflow as tf
import cv2
import glob
from pathlib import Path


# Configuration
TFLITE_MODEL_PATH = "./models/model_optimized.tflite"
SYMBOLS_FILE = "./symbols.txt"
VALIDATION_DIR = "./validation"
OUTPUT_CSV = "./validation_predictions_tflite.csv"
IMG_WIDTH = 192
IMG_HEIGHT = 96
PADDING_CHAR = '_'


class TFLiteCaptchaPredictor:
    """TFLite-based CAPTCHA predictor with fixed output handling"""
    
    def __init__(self, tflite_model_path, symbols_file):
        self.tflite_model_path = tflite_model_path
        self.symbols_file = symbols_file
        
        # Load symbols
        self.symbols = self._load_symbols()
        self.num_to_char = {i: c for i, c in enumerate(self.symbols)}
        
        # Load TFLite model
        self.interpreter = self._load_model()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Sort output details by name to ensure correct order (char_0, char_1, ...)
        self.output_details = sorted(self.output_details, key=lambda x: x['name'])
        
        print(f"Loaded TFLite model: {os.path.basename(tflite_model_path)}")
        print(f"Symbols: {len(self.symbols)} characters")
        print(f"Output heads: {len(self.output_details)}")
        print(f"Output order: {[o['name'] for o in self.output_details]}")
    
    def _load_symbols(self):
        """Load symbol set from file"""
        if not os.path.exists(self.symbols_file):
            raise FileNotFoundError(f"Symbols file not found: {self.symbols_file}")
        
        with open(self.symbols_file, 'r') as f:
            base_symbols = f.readline().strip()
        
        # Add padding character
        return base_symbols + PADDING_CHAR
    
    def _load_model(self):
        """Load TFLite model"""
        if not os.path.exists(self.tflite_model_path):
            raise FileNotFoundError(f"TFLite model not found: {self.tflite_model_path}")
        
        interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    def preprocess_image(self, img_path):
        """
        Preprocess image with EXACT same steps as training
        """
        # Load image as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Resize to model input size
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        # Bilateral filter - denoises while preserving edges
        img = cv2.bilateralFilter(img, 5, 75, 75)
        
        # CLAHE - improves contrast adaptively
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add channel dimension (grayscale = 1 channel)
        img = np.expand_dims(img, axis=-1)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, img):
        """
        Run inference on preprocessed image
        """
        # Ensure input is float32
        img = img.astype(np.float32)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get all outputs IN CORRECT ORDER
        outputs = []
        for output_detail in self.output_details:
            output = self.interpreter.get_tensor(output_detail['index'])
            outputs.append(output)
        
        return outputs
    
    def decode_prediction(self, outputs, stop_at_padding=True):
        """
        Decode model outputs to text and confidence
        
        Args:
            outputs: List of output arrays from model
            stop_at_padding: Whether to stop at first padding character
        
        Returns:
            tuple: (predicted_text, average_confidence, all_chars_with_conf)
        """
        predicted_chars = []
        confidences = []
        all_predictions = []
        
        for i, output in enumerate(outputs):
            # Get the character with highest probability
            char_idx = np.argmax(output[0])
            confidence = float(output[0][char_idx])
            
            # Convert index to character
            char = self.num_to_char.get(char_idx, '?')
            
            all_predictions.append({
                'position': i,
                'char': char,
                'confidence': confidence,
                'char_idx': char_idx
            })
            
            # Stop at padding character if requested
            if stop_at_padding and char == PADDING_CHAR:
                break
            
            # Only add non-padding characters
            if char != PADDING_CHAR:
                predicted_chars.append(char)
                confidences.append(confidence)
        
        predicted_text = ''.join(predicted_chars)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        
        return predicted_text, avg_confidence, all_predictions
    
    def predict_image(self, img_path, debug=False):
        """
        Complete prediction pipeline for a single image
        
        Args:
            img_path: Path to image file
            debug: If True, return detailed predictions
        
        Returns:
            tuple: (predicted_text, confidence) or (text, conf, details) if debug=True
        """
        # Preprocess
        img = self.preprocess_image(img_path)
        
        # Predict
        outputs = self.predict(img)
        
        # Decode
        text, confidence, details = self.decode_prediction(outputs, stop_at_padding=True)
        
        # If empty, try without stopping at padding
        if not text:
            text_no_stop, confidence_no_stop, details_no_stop = self.decode_prediction(outputs, stop_at_padding=False)
            if text_no_stop:  # If we got something without stopping
                text = text_no_stop
                confidence = confidence_no_stop
                details = details_no_stop
        
        if debug:
            return text, confidence, details
        else:
            return text, confidence


def find_validation_images(validation_dir):
    """Find all image files in validation directory"""
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.PNG', '*.JPG', '*.JPEG']
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(validation_dir, ext)))
    
    return sorted(image_files)


def predict_captchas(predictor, validation_dir, show_first_n_details=5):
    """
    Run predictions on all images in validation directory
    """
    # Find all images
    image_files = find_validation_images(validation_dir)
    
    if not image_files:
        print(f"No images found in {validation_dir}")
        return None
    
 
    print(f"Predicting {len(image_files)} CAPTCHAs")
    
    results = []
    
    # Show detailed output for first few images
    for i, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        
        try:
            # Predict (with debug info for first few)
            if i <= show_first_n_details:
                text, confidence, details = predictor.predict_image(img_path, debug=True)
                
                # Show detailed breakdown
                print(f"\n[{i:3d}/{len(image_files)}] {filename}")
                print(f"  Position breakdown:")
                for d in details:
                    print(f"    Pos {d['position']}: '{d['char']}' (conf={d['confidence']:.4f}, idx={d['char_idx']})")
                print(f"  Final: '{text}' (avg_conf={confidence:.4f})")
            else:
                text, confidence = predictor.predict_image(img_path, debug=False)
                
                # Regular output
                if i % 100 == 0 or i <= 20:  # Show progress
                    print(f"[{i:3d}/{len(image_files)}] {filename:40s} → {text:15s} (conf={confidence:.3f})")
            
            # Store result
            results.append({
                'filename': filename,
                'prediction': text,
                'confidence': confidence
            })
        
        except Exception as e:
            print(f"[{i:3d}/{len(image_files)}] {filename:40s} → ERROR: {str(e)}")
            results.append({
                'filename': filename,
                'prediction': 'ERROR',
                'confidence': 0.0
            })
    
    return results


def save_results_to_csv(results, output_path, username="dheenadh"):
    """
    Save prediction results to CSV file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write username in first line
        f.write(f"{username}\n")
        
        # Write predictions
        for result in results:
            f.write(f"{result['filename']},{result['prediction']}\n")
    
    print(f"\nResults saved to: {output_path}")


def print_summary(results):
    """Print prediction summary statistics"""
    total = len(results)
    errors = sum(1 for r in results if r['prediction'] == 'ERROR')
    empty = sum(1 for r in results if r['prediction'] == '' and r['prediction'] != 'ERROR')
    successful = total - errors - empty
    
    if successful > 0:
        avg_conf = np.mean([r['confidence'] for r in results if r['prediction'] not in ['ERROR', '']])
    else:
        avg_conf = 0.0
    
    print(f"PREDICTION SUMMARY")

    print(f"Total images: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Empty predictions: {empty} ({empty/total*100:.1f}%)")
    print(f"Errors: {errors} ({errors/total*100:.1f}%)")
    print(f"Average confidence: {avg_conf:.3f}")
    
    if empty > total * 0.5:
        print(f"\nWARNING: More than 50% empty predictions!")
        print(f"   This suggests the TFLite model may have conversion issues.")
        print(f"   Please run the diagnostic script to investigate.")
    
    print(f"{'='*70}\n")


def main():
    """Main prediction pipeline"""
    
    print("CAPTCHA Prediction")
    
    # Check files exist
    if not os.path.exists(TFLITE_MODEL_PATH):
        print(f"Error: TFLite model not found at {TFLITE_MODEL_PATH}")
        print("Please run the H5 to TFLite converter first.")
        return
    
    if not os.path.exists(SYMBOLS_FILE):
        print(f"Error: Symbols file not found at {SYMBOLS_FILE}")
        return
    
    if not os.path.exists(VALIDATION_DIR):
        os.makedirs(VALIDATION_DIR)
        print(f"Created validation directory: {VALIDATION_DIR}")
        print("Add images to this directory and run again.")
        return
    
    # Initialize predictor
    print("Loading TFLite model...")
    try:
        predictor = TFLiteCaptchaPredictor(TFLITE_MODEL_PATH, SYMBOLS_FILE)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run predictions
    results = predict_captchas(predictor, VALIDATION_DIR, show_first_n_details=5)
    
    if results is None:
        return
    
    # Save results
    save_results_to_csv(results, OUTPUT_CSV)
    
    # Print summary
    print_summary(results)
    
    print("Prediction complete!\n")


if __name__ == "__main__":
    main()
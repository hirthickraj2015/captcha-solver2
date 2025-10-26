#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy as np
import argparse
import tflite_runtime.interpreter as tflite

def decode(characters, y):
    """
    Converts TFLite model predictions to string.
    Supports both single-output (one tensor with multiple chars) 
    and multi-output (one tensor per character) models.
    """
    result = ''

    # Detect if y is list of arrays (multi-output) or single array (single-output)
    if isinstance(y, list):
        # Multi-output: one tensor per character
        for char_pred in y:
            char = characters[np.argmax(char_pred)]
            if char != '?':
                result += char
    else:
        # Single-output: tensor shape (1, captcha_length, num_symbols)
        y = np.argmax(y, axis=2)[0]  # Take first batch
        for idx in y:
            if characters[idx] != '?':
                result += characters[idx]

    return result

def preprocess_image(img_path):
    """
    Read, resize, normalize and reshape the image for TFLite model.
    """
    raw_data = cv2.imread(img_path)
    if raw_data is None:
        return None
    if raw_data.shape[:2] != (96, 192):
        raw_data = cv2.resize(raw_data, (192, 96))
    rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
    image = np.array(rgb_data, dtype=np.float32) / 255.0
    image = image.reshape([1, 96, 192, 3])
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, help='TFLite model file (.tflite)')
    parser.add_argument('--captcha-dir', type=str, help='Directory with captchas to classify')
    parser.add_argument('--output', type=str, help='File where classifications will be saved')
    parser.add_argument('--symbols', type=str, help='File with captcha symbols')
    args = parser.parse_args()

    # Validate arguments
    for arg_name in ['model_name', 'captcha_dir', 'output', 'symbols']:
        if getattr(args, arg_name) is None:
            print(f"Please specify {arg_name.replace('_',' ')}")
            exit(1)

    # Load captcha symbols
    with open(args.symbols, 'r', encoding='utf-8') as f:
        captcha_symbols = f.readline().strip('\n')

    print(f"Classifying captchas with symbol set {{{captcha_symbols}}}")
    print(f"Total symbols: {len(captcha_symbols)}")

    # Load TFLite model
    interpreter = tflite.Interpreter(model_path=args.model_name)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Determine if single-output or multi-output model
    multi_output = len(output_details) > 1
    if multi_output:
        print(f"Detected multi-output TFLite model with {len(output_details)} outputs")
    else:
        print("Detected single-output TFLite model")

    # Open output file
    with open(args.output, 'w') as output_file:
        output_file.write("dheenadh\n")
        idx = 1

        for filename in sorted(os.listdir(args.captcha_dir)):
            if filename.startswith('.'):
                continue
            img_path = os.path.join(args.captcha_dir, filename)
            if not os.path.isfile(img_path):
                continue

            try:
                image = preprocess_image(img_path)
                if image is None:
                    print(f"Warning: Could not read {filename}, skipping...")
                    continue

                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], image)
                interpreter.invoke()

                # Collect predictions
                if multi_output:
                    preds = [interpreter.get_tensor(out['index'])[0] for out in output_details]
                else:
                    preds = interpreter.get_tensor(output_details[0]['index'])

                captcha_str = decode(captcha_symbols, preds)

                output_file.write(f"{filename},{captcha_str}\n")
                print(f"Classified: {filename} -> {captcha_str}; Progress: {idx}")
                idx += 1

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    print(f"\nClassification complete! Results saved to {args.output}")

if __name__ == '__main__':
    main()
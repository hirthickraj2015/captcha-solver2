#!/usr/bin/env python3

import os
import numpy as np
import random
import string
import cv2
import argparse
import captcha.image
import hashlib
import csv
from sklearn.model_selection import train_test_split

def safe_filename(text: str, ext=".png"):
    encoded = text.encode('utf-8')
    hash_str = hashlib.md5(encoded).hexdigest()
    return hash_str + ext

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, required=True, help='Width of captcha image')
    parser.add_argument('--height', type=int, required=True, help='Height of captcha image')
    parser.add_argument('--length', type=int, required=True, help='Length of captchas in characters')
    parser.add_argument('--count', type=int, required=True, help='Number of captchas per font')
    parser.add_argument('--output-dir', type=str, required=True, help='Dataset folder')
    parser.add_argument('--symbols', type=str, required=True, help='File with symbols to use in captchas')
    parser.add_argument('--fonts-dir', type=str, default='fonts', help='Folder containing fonts')
    args = parser.parse_args()

    # Load fonts
    font_paths = [os.path.join(args.fonts_dir, f) for f in os.listdir(args.fonts_dir)
                  if f.lower().endswith(('.ttf', '.otf'))]
    if not font_paths:
        print(f"No fonts found in {args.fonts_dir}. Exiting.")
        exit(1)

    # Load symbols
    with open(args.symbols, 'r', encoding='utf-8') as f:
        captcha_symbols = f.readline().strip()
    valid_symbols = captcha_symbols.replace('?', '')

    print(f"Generating captchas with symbols: {captcha_symbols}")
    
    # Prepare output folders
    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Initialize CSV writers
    train_csv = open(os.path.join(train_dir, "labels.csv"), 'w', newline='', encoding='utf-8')
    test_csv = open(os.path.join(test_dir, "labels.csv"), 'w', newline='', encoding='utf-8')
    train_writer = csv.writer(train_csv)
    test_writer = csv.writer(test_csv)
    train_writer.writerow(["filename", "label"])
    test_writer.writerow(["filename", "label"])

    for font_path in font_paths:
        print(f"Generating captchas for font: {os.path.basename(font_path)}")
        captcha_generator = captcha.image.ImageCaptcha(width=args.width, height=args.height, fonts=[font_path])

        data = []
        for _ in range(args.count):
            valid_length = random.randint(1, args.length)
            random_str = ''.join([random.choice(valid_symbols) for _ in range(valid_length)])
            filename_str = random_str + '?' * (args.length - valid_length)
            hash_name = hashlib.md5((random_str + os.path.basename(font_path)).encode('utf-8')).hexdigest()
            data.append((hash_name, filename_str, random_str))

        # Train/test split 80/20
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        # Save images and CSVs
        for hash_name, filename_str, random_str in train_data:
            image_path = os.path.join(train_dir, hash_name + '.png')
            image = np.array(captcha_generator.generate_image(random_str))
            cv2.imwrite(image_path, image)
            train_writer.writerow([hash_name, filename_str])

        for hash_name, filename_str, random_str in test_data:
            image_path = os.path.join(test_dir, hash_name + '.png')
            image = np.array(captcha_generator.generate_image(random_str))
            cv2.imwrite(image_path, image)
            test_writer.writerow([hash_name, filename_str])

    train_csv.close()
    test_csv.close()
    print("Captcha generation completed.")

if __name__ == '__main__':
    main()

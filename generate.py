#!/usr/bin/env python3
import os
import numpy
import random
import cv2
import hashlib
import csv
from captcha.image import ImageCaptcha



# configration

font_dir = "./fonts"                  
output_dir = "./dataset"               
symbols_file = "./symbols.txt"         
use_mixed_fonts = False               
image_width = 192                     
image_height = 96                      
min_length = 1                         
max_length = 6                         
captchas_per_font = 100000             
train_ratio = 0.8                     



def generate_hashed_filename(text: str) -> str:
    """Generate a SHA1 hash for the given captcha text."""
    return hashlib.sha1(text.encode()).hexdigest() + ".png"


def save_label_csv(csv_path: str, rows: list):
    """Save list of (filename, label) to CSV."""
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)


def generate_captchas_for_font(font_name, font_path, symbols):
    """Generate captchas for a specific font, split into train/test, and save labels."""
    font_base_name = os.path.splitext(font_name)[0]
    print(f"Generating captchas for font: {font_base_name}")

    # Create generator
    generator = ImageCaptcha(width=image_width, height=image_height, fonts=[font_path])
    generator.character_warp_dx = (0.1, 0.5)
    generator.character_warp_dy = (0.2, 0.5)
    generator.character_rotate = (-45, 45)

    # Prepare folder structure
    font_train_dir = os.path.join(output_dir, "train", font_base_name, "images")
    font_test_dir = os.path.join(output_dir, "test", font_base_name, "images")
    os.makedirs(font_train_dir, exist_ok=True)
    os.makedirs(font_test_dir, exist_ok=True)

    # For CSV mappings
    train_labels = []
    test_labels = []

    for i in range(captchas_per_font):
        length = random.randint(min_length, max_length)
        captcha_text = ''.join(random.choice(symbols) for _ in range(length))
        hashed_name = generate_hashed_filename(captcha_text)
        image_array = numpy.array(generator.generate_image(captcha_text))

        # Decide train/test split
        if random.random() < train_ratio:
            image_path = os.path.join(font_train_dir, hashed_name)
            train_labels.append((hashed_name, captcha_text))
        else:
            image_path = os.path.join(font_test_dir, hashed_name)
            test_labels.append((hashed_name, captcha_text))

        cv2.imwrite(image_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

    # Save label files
    save_label_csv(os.path.join(output_dir, "train", font_base_name, "labels.csv"), train_labels)
    save_label_csv(os.path.join(output_dir, "test", font_base_name, "labels.csv"), test_labels)

    print(f"   {len(train_labels)} train images, {len(test_labels)} test images saved for {font_base_name}")


def main():
    # Load symbol set
    with open(symbols_file, 'r') as f:
        captcha_symbols = f.readline().strip()

    print(f"Generating captchas with symbol set: {captcha_symbols}")
    os.makedirs(output_dir, exist_ok=True)

    # Load fonts
    font_names = sorted(os.listdir(font_dir))
    font_paths = [os.path.join(font_dir, f) for f in font_names if os.path.isfile(os.path.join(font_dir, f))]
    print(f"Found {len(font_names)} fonts")

    if use_mixed_fonts:
        print("Mixed font mode not implemented with hashing yet.")
        return

    # Generate captchas per font
    for font_name, font_path in zip(font_names, font_paths):
        generate_captchas_for_font(font_name, font_path, captcha_symbols)

    print("\nDataset generation complete!")
    
if __name__ == "__main__":
    main()

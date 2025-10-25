import os
import cv2
import random
from pathlib import Path

# CONFIGURATION
DATA_DIR = "./dataset"          
PREVIEW_DIR = "./previews"      
NUM_SAMPLES = 10                
IMAGE_WIDTH = 192
IMAGE_HEIGHT = 96

os.makedirs(PREVIEW_DIR, exist_ok=True)


# PREPROCESS FUNCTION
def preprocess_image(img_path):
    """Resize, denoise, and apply CLAHE — same as training pipeline."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f" Skipping unreadable image: {img_path}")
        return None

    # Resize for consistency
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Denoise and enhance
    img = cv2.bilateralFilter(img, 5, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    return img



if __name__ == "__main__":
    image_paths = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {DATA_DIR} or its subfolders.")

    selected = random.sample(image_paths, min(NUM_SAMPLES, len(image_paths)))
    print(f"Processing {len(selected)} sample images for preview...")

    # Process and save previews
    for i, img_path in enumerate(selected, 1):  
        processed = preprocess_image(img_path)
        if processed is not None:
            out_path = os.path.join(PREVIEW_DIR, f"preview_{i}_{Path(img_path).stem}.png")
            cv2.imwrite(out_path, processed)
            print(f"Saved: {out_path}")

    print("\nDone! './previews' folder for has samples.")

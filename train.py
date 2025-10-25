#!/usr/bin/env python3
import os
import glob
import json
import csv
import shutil
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

# configuration
DATA_DIR = "./dataset"
SYMBOLS_FILE = "./symbols.txt"
OUTPUT_DIR = "./models"
IMG_WIDTH = 192
IMG_HEIGHT = 96
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MIXED_PRECISION = False 

# Augmentation probabilities
AUG_PROBS = {
    "rotate": 0.5,
    "perspective": 0.3,
    "elastic": 0.25,
    "illum": 0.4,
    "blur": 0.3,
    "add_lines": 0.5
}


if MIXED_PRECISION:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)


def find_latest_checkpoint(output_dir):
    pattern = os.path.join(output_dir, 'checkpoint_epoch_*.h5')
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None, 0
    epochs = []
    for c in ckpts:
        try:
            e = int(os.path.basename(c).split('_epoch_')[1].split('.h5')[0])
            epochs.append((e, c))
        except Exception:
            continue
    if not epochs:
        return None, 0
    epochs.sort(reverse=True)
    return epochs[0][1], epochs[0][0]


def load_data_from_csv(data_dir, symbols):
    image_paths, labels = [], []
    print(f"Searching for data in: {data_dir}")
    for root, _, files in os.walk(data_dir):
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
                        if not all(c in symbols for c in label):
                            continue
                        img_path = os.path.join(images_dir, filename)
                        if os.path.exists(img_path):
                            image_paths.append(img_path)
                            labels.append(label)
    print(f"Total images: {len(image_paths):,}")
    if labels:
        dist = {}
        for l in labels:
            dist[len(l)] = dist.get(len(l), 0) + 1
        print("Label length distribution:")
        for k in sorted(dist.keys()):
            pct = dist[k]/len(labels)*100
            print(f"  {k}: {dist[k]:,} ({pct:.1f}%)")
    return image_paths, labels


# Augmentation

def random_illumination(img, strength=0.25):
    # brightness and contrast jitter
    alpha = 1.0 + (np.random.uniform(-strength, strength))
    beta = np.random.uniform(-30, 30)
    res = img * alpha + beta
    res = np.clip(res, 0, 255).astype(np.uint8)
    return res


def random_perspective(img, max_warp=0.08):
    h, w = img.shape[:2]
    margin_x = int(w * max_warp)
    margin_y = int(h * max_warp)
    pts1 = np.float32([[0,0],[w,0],[w,h],[0,h]])
    pts2 = np.float32([
        [np.random.randint(0, margin_x), np.random.randint(0, margin_y)],
        [w - np.random.randint(0, margin_x), np.random.randint(0, margin_y)],
        [w - np.random.randint(0, margin_x), h - np.random.randint(0, margin_y)],
        [np.random.randint(0, margin_x), h - np.random.randint(0, margin_y)]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def elastic_transform(image, alpha, sigma):
    # Simple elastic transform implementation
    shape = image.shape
    dx = (np.random.rand(*shape) * 2 - 1)
    dy = (np.random.rand(*shape) * 2 - 1)
    dx = cv2.GaussianBlur(dx, (17,17), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (17,17), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def add_noise_lines(img, n_lines=2):
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(n_lines):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        thickness = np.random.randint(1, 3)
        color = int(np.random.uniform(0, 255))
        cv2.line(out, (x1,y1), (x2,y2), color, thickness)
    return out


# ---------------- Data Generator ----------------
class CaptchaDataGenerator(keras.utils.Sequence):
    def __init__(self, image_paths, labels, symbols, img_width, img_height,
                 max_length, batch_size=32, shuffle=True, is_training=False):
        self.image_paths = image_paths
        self.labels = labels
        self.symbols = symbols
        self.char_to_num = {c: i for i,c in enumerate(symbols)}
        self.img_w = img_width
        self.img_h = img_height
        self.max_len = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_training = is_training
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        # include last partial batch
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def preprocess_captcha(self, img):
        # input img is grayscale uint8
        # Basic denoise while preserving edges
        img = cv2.bilateralFilter(img, 5, 75, 75)
        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        return img

    def augment(self, img):
        # img: uint8 grayscale
        if random.random() < AUG_PROBS['rotate']:
            angle = np.random.uniform(-18, 18)
            M = cv2.getRotationMatrix2D((self.img_w/2, self.img_h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (self.img_w, self.img_h), borderMode=cv2.BORDER_REPLICATE)
        if random.random() < AUG_PROBS['perspective']:
            img = random_perspective(img)
        if random.random() < AUG_PROBS['elastic']:
            img = elastic_transform(img, alpha=1.5, sigma=8)
        if random.random() < AUG_PROBS['illum']:
            img = random_illumination(img, strength=0.25)
        if random.random() < AUG_PROBS['blur']:
            k = random.choice([3,5])
            img = cv2.GaussianBlur(img, (k,k), 0)
        if random.random() < AUG_PROBS['add_lines']:
            img = add_noise_lines(img, n_lines=random.randint(0,3))
        return img

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size: (idx+1) * self.batch_size]
        batch_paths = [self.image_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]

        X = np.zeros((len(batch_paths), self.img_h, self.img_w, 1), dtype=np.float32)
        y = {f'char_{i}': np.zeros((len(batch_paths),), dtype=np.int32) for i in range(self.max_len)}

        for i, (p, label) in enumerate(zip(batch_paths, batch_labels)):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
            else:
                img = cv2.resize(img, (self.img_w, self.img_h))

            # preprocessing
            img = self.preprocess_captcha(img)

            # augmentation
            if self.is_training:
                img = self.augment(img)

            # normalize to 0..1
            img = img.astype(np.float32) / 255.0
            X[i,:,:,0] = img

            # encode labels
            for j in range(self.max_len):
                if j < len(label):
                    ch = label[j]
                else:
                    ch = '_'
                y[f'char_{j}'][i] = self.char_to_num.get(ch, self.char_to_num.get('_', 0))

        return X, y


#  Model 

def build_model(img_w, img_h, max_length, num_classes):
    inputs = keras.Input(shape=(img_h, img_w, 1))
    reg = regularizers.l2(WEIGHT_DECAY)

    def conv_block(x, filters):
        x = keras.layers.Conv2D(filters, (3,3), padding='same', kernel_regularizer=reg)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(filters, (3,3), padding='same', kernel_regularizer=reg)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D((2,2))(x)
        x = keras.layers.Dropout(0.2)(x)
        return x

    x = conv_block(inputs, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation='relu', kernel_regularizer=reg)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)

    # outputs
    outputs = [keras.layers.Dense(num_classes, activation='softmax', name=f'char_{i}')(x)
               for i in range(max_length)]

    model = keras.Model(inputs, outputs)
    return model


# Training 
class EpochCheckpoint(keras.callbacks.Callback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
    def on_epoch_end(self, epoch, logs=None):
        path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.h5')
        self.model.save(path)
        print(f"Saved: {os.path.basename(path)}")


def load_training_history(output_dir):
    path = os.path.join(output_dir, 'training_history.csv')
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"Loaded training history: {len(df)} epochs")
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_training_history(output_dir, history_dict, start_epoch=0):
    existing = load_training_history(output_dir)
    new = {k: v for k, v in history_dict.items()}
    num_epochs = len(next(iter(new.values())))
    new['epoch'] = list(range(start_epoch+1, start_epoch + num_epochs + 1))
    df_new = pd.DataFrame(new)
    if not existing.empty:
        out = pd.concat([existing, df_new], ignore_index=True)
    else:
        out = df_new
    out.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    print('✓ Training history saved')
    return out


def find_best_model_from_history(output_dir):
    path = os.path.join(output_dir, 'training_history.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'val_loss' not in df.columns:
        return None
    best_idx = df['val_loss'].idxmin()
    best_epoch = int(df.loc[best_idx, 'epoch'])
    best_ckpt = os.path.join(output_dir, f'checkpoint_epoch_{best_epoch}.h5')
    if os.path.exists(best_ckpt):
        final = os.path.join(output_dir, 'best_model_final.h5')
        shutil.copy2(best_ckpt, final)
        print(f'✓ Best model copied: {final}')
        return best_ckpt
    return None


# Main training flow 

def train_model():
    with open(SYMBOLS_FILE, 'r') as f:
        base_symbols = f.readline().strip()
    symbols = base_symbols + '_'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    latest_checkpoint, start_epoch = find_latest_checkpoint(OUTPUT_DIR)
    if latest_checkpoint:
        print(f"Found checkpoint from epoch {start_epoch}: {latest_checkpoint}")
        # keep behavior: ask user whether to resume
        resume = input('Resume training? (y/n): ').strip().lower()
        if resume != 'y':
            latest_checkpoint = None
            start_epoch = 0
    else:
        start_epoch = 0

    image_paths, labels = load_data_from_csv(DATA_DIR, base_symbols)
    if not image_paths:
        print('No data found!')
        return

    max_length = max(len(l) for l in labels)
    print(f"Max CAPTCHA length: {max_length}")

    # train/val split
    from sklearn.model_selection import train_test_split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, shuffle=True)

    train_gen = CaptchaDataGenerator(train_paths, train_labels, symbols, IMG_WIDTH, IMG_HEIGHT,
                                     max_length, batch_size=BATCH_SIZE, shuffle=True, is_training=True)
    val_gen = CaptchaDataGenerator(val_paths, val_labels, symbols, IMG_WIDTH, IMG_HEIGHT,
                                   max_length, batch_size=BATCH_SIZE, shuffle=False, is_training=False)

    if latest_checkpoint:
        model = keras.models.load_model(latest_checkpoint)
    else:
        model = build_model(IMG_WIDTH, IMG_HEIGHT, max_length, len(symbols))

        opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
        model.compile(optimizer=opt,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    model.summary()

    callbacks = [
        EpochCheckpoint(OUTPUT_DIR),
        keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR, 'best_model_ongoing.h5'),
                                        save_best_only=True, monitor='val_loss', verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        keras.callbacks.CSVLogger(os.path.join(OUTPUT_DIR, 'training_log.csv'), append=True)
    ]

    print(f"Training: {len(train_paths):,}  Validation: {len(val_paths):,}  Steps/epoch: {len(train_paths)//BATCH_SIZE}")

    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=start_epoch + EPOCHS,
                        initial_epoch=start_epoch,
                        callbacks=callbacks,
                        verbose=1)

    final_model = os.path.join(OUTPUT_DIR, 'model_final.h5')
    model.save(final_model)
    print(f"Final model saved: {final_model}")

    save_training_history(OUTPUT_DIR, history.history, start_epoch)
    find_best_model_from_history(OUTPUT_DIR)
    print('Training complete')


if __name__ == '__main__':
    train_model()

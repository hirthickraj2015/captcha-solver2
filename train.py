#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy as np
import argparse
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def decode(characters, y):
    """
    Converts model predictions to string based on character set.
    Handles multi-output model format.
    """
    if isinstance(y, list):
        # Multi-output format: list of predictions for each character
        result = ''
        for char_pred in y:
            char = characters[np.argmax(char_pred[0])]
            if char != '?':
                result += char
        return result
    else:
        # Single output format
        y = np.argmax(np.array(y), axis=2)[:, 0]
        return ''.join([characters[x] for x in y if characters[x] != '?'])

def build_model_from_json(captcha_length=6, captcha_num_symbols=43):
    """
    Build the exact model architecture from the JSON specification.
    Input shape: (96, 192, 3)
    """
    # Input layer
    inputs = Input(shape=(96, 192, 3), name='input_layer')
    
    # First Conv Block (32 filters)
    x = Conv2D(32, (3, 3), padding='same', use_bias=True, name='conv2d')(inputs)
    x = BatchNormalization(name='batch_normalization')(x)
    x = Activation('relu', name='activation')(x)
    
    x = Conv2D(32, (3, 3), padding='same', use_bias=True, name='conv2d_1')(x)
    x = BatchNormalization(name='batch_normalization_1')(x)
    x = Activation('relu', name='activation_1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d')(x)
    
    # Second Conv Block (64 filters)
    x = Conv2D(64, (3, 3), padding='same', use_bias=True, name='conv2d_2')(x)
    x = BatchNormalization(name='batch_normalization_2')(x)
    x = Activation('relu', name='activation_2')(x)
    
    x = Conv2D(64, (3, 3), padding='same', use_bias=True, name='conv2d_3')(x)
    x = BatchNormalization(name='batch_normalization_3')(x)
    x = Activation('relu', name='activation_3')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1')(x)
    
    # Third Conv Block (128 filters)
    x = Conv2D(128, (3, 3), padding='same', use_bias=True, name='conv2d_4')(x)
    x = BatchNormalization(name='batch_normalization_4')(x)
    x = Activation('relu', name='activation_4')(x)
    
    x = Conv2D(128, (3, 3), padding='same', use_bias=True, name='conv2d_5')(x)
    x = BatchNormalization(name='batch_normalization_5')(x)
    x = Activation('relu', name='activation_5')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2')(x)
    
    # Fourth Conv Block (256 filters)
    x = Conv2D(256, (3, 3), padding='same', use_bias=True, name='conv2d_6')(x)
    x = BatchNormalization(name='batch_normalization_6')(x)
    x = Activation('relu', name='activation_6')(x)
    
    x = Conv2D(256, (3, 3), padding='same', use_bias=True, name='conv2d_7')(x)
    x = BatchNormalization(name='batch_normalization_7')(x)
    x = Activation('relu', name='activation_7')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_3')(x)
    
    # Fifth Conv Block (256 filters)
    x = Conv2D(256, (3, 3), padding='same', use_bias=True, name='conv2d_8')(x)
    x = BatchNormalization(name='batch_normalization_8')(x)
    x = Activation('relu', name='activation_8')(x)
    
    x = Conv2D(256, (3, 3), padding='same', use_bias=True, name='conv2d_9')(x)
    x = BatchNormalization(name='batch_normalization_9')(x)
    x = Activation('relu', name='activation_9')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_4')(x)
    
    # Flatten
    x = Flatten(name='flatten')(x)
    
    # Output layers - one Dense layer per character position
    outputs = []
    for i in range(1, captcha_length + 1):
        output = Dense(captcha_num_symbols, activation='softmax', name=f'char_{i}')(x)
        outputs.append(output)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, help='Model name to use for classification')
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

    captcha_num_symbols = len(captcha_symbols)
    captcha_length = 6  # From the JSON structure
    
    print(f"Classifying captchas with symbol set {{{captcha_symbols}}}")
    print(f"Total symbols: {captcha_num_symbols}, CAPTCHA length: {captcha_length}")

    with tf.device('/cpu:0'):
        # Build the model architecture
        print("Building model architecture...")
        model = build_model_from_json(captcha_length, captcha_num_symbols)
        
        try:
            # Load weights
            print("Loading weights...")
            model.load_weights(args.model_name + '.h5')
            print("Successfully loaded weights!")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("\nMake sure the model file exists at:", args.model_name + '.h5')
            exit(1)
        
        # Compile the model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
            metrics=['accuracy']
        )

        # Open output file for writing
        with open(args.output, 'w') as output_file:
            idx = 1
            for filename in sorted(os.listdir(args.captcha_dir)):
                # Skip hidden files
                if filename.startswith('.'):
                    continue
                    
                img_path = os.path.join(args.captcha_dir, filename)
                
                # Skip if not a file
                if not os.path.isfile(img_path):
                    continue
                
                try:
                    # Read and preprocess image
                    raw_data = cv2.imread(img_path)
                    if raw_data is None:
                        print(f"Warning: Could not read {filename}, skipping...")
                        continue
                    
                    # Resize to expected input shape (96, 192)
                    if raw_data.shape[:2] != (96, 192):
                        raw_data = cv2.resize(raw_data, (192, 96))
                    
                    rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                    image = np.array(rgb_data) / 255.0
                    image = image.reshape([1, 96, 192, 3])

                    # Predict captcha
                    pred = model.predict(image, verbose=0)
                    captcha_str = decode(captcha_symbols, pred)
                    
                    output_file.write(f"{filename}, {captcha_str}\n")
                    print(f"Classified: {filename} -> {captcha_str}; Progress: {idx}")
                    idx += 1
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
        
        print(f"\nClassification complete! Results saved to {args.output}")

if __name__ == '__main__':
    main()
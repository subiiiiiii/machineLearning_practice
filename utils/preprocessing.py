# utils/preprocessing.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 32))  # Resize to match model input
    img = img / 255.0  # Normalize
    img = img.reshape(1, 32, 128, 1)  # Reshape to (1, 32, 128, 1) for single image
    return img

def load_and_preprocess_data():
    X = []
    y = []

    processed_dir = '../data/processed/'
    for filename in os.listdir(processed_dir):
        if filename.endswith('.npy'):
            img = np.load(os.path.join(processed_dir, filename))
            X.append(img)
            label = get_label(filename)  # Fetch or generate label here
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)

    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_val, y_val

def get_label(filename):
    label = [0] * 36  # Assuming 36 classes
    label[ord(filename[0]) % 36] = 1  # Simple example: one-hot encoding based on filename
    return label

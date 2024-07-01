import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Directories
processed_data_dir = '../data/processed/'

def preprocess_image(image):
    if isinstance(image, str):  # If a path is provided, load the image
        if not os.path.exists(image):
            raise ValueError(f"File not found: {image}")
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image}")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(image, (128, 32))  # Resize to match model input
    img = img / 255.0  # Normalize
    img = img.reshape(1, 32, 128, 1)  # Reshape to (1, 32, 128, 1) for single image
    return img

def process_and_save_images(raw_data_dir):
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    for filename in os.listdir(raw_data_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(raw_data_dir, filename)
            print(f"Processing file: {filename}")  # Debugging statement
            try:
                img = preprocess_image(image_path)
                np.save(os.path.join(processed_data_dir, os.path.splitext(filename)[0] + '.npy'), img)
                print(f'Processed and saved: {filename}')
            except ValueError as e:
                print(e)

def load_and_preprocess_data():
    X = []
    y = []

    for filename in os.listdir(processed_data_dir):
        if filename.endswith('.npy'):
            img = np.load(os.path.join(processed_data_dir, filename))
            X.append(img)
            label = get_label(filename)  # Fetch or generate label here
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)

    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')
    print(f'Sample X: {X[0]}')
    print(f'Sample y: {y[0]}')

    X = X.reshape((X.shape[0], 32, 128, 1))

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_val, y_val

def get_label(filename):
    label = np.zeros(36)  # Assuming 36 classes
    label[ord(filename[0]) % 36] = 1  # Simple example: one-hot encoding based on filename
    return label

if __name__ == "__main__":
    raw_data_dir = '../data/raw/'
    print(f"Current working directory: {os.getcwd()}")
    process_and_save_images(raw_data_dir)

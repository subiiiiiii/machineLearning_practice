import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from preprocessing import load_and_preprocess_data

# Load and preprocess data
X_train, y_train, X_val, y_val = load_and_preprocess_data()

# Ensure data is loaded correctly
assert X_train.size > 0, "X_train is empty!"
assert y_train.size > 0, "y_train is empty!"
assert X_val.size > 0, "X_val is empty!"
assert y_val.size > 0, "y_val is empty!"

# Define the model
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

input_shape = (32, 128, 1)
num_classes = 36  # 26 letters + 10 digits
model = build_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save('model.h5')

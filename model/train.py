import sys
import os

# Add the utils directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from preprocessing import load_and_preprocess_data

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load and preprocess data
X_train, y_train, X_val, y_val = load_and_preprocess_data()

# Ensure data is loaded correctly
assert X_train.size > 0, "X_train is empty!"
assert y_train.size > 0, "y_train is empty!"
assert X_val.size > 0, "X_val is empty!"
assert y_val.size > 0, "y_val is empty!"

# Define the model
def build_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Reshape((-1, 128)))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

input_shape = (32, 128, 1)
num_classes = 36  # 26 letters + 10 digits
model = build_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save('model.h5')

# utils/postprocessing.py

import numpy as np

CHARACTER_SET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def decode_prediction(prediction):
    # Assuming prediction is of shape (batch_size, timesteps, num_classes)
    char_indices = np.argmax(prediction, axis=-1)
    decoded_text = ''.join(CHARACTER_SET[idx] for idx in char_indices.flatten() if idx < len(CHARACTER_SET))
    return decoded_text

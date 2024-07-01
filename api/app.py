import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from preprocessing import preprocess_image
from postprocessing import decode_prediction

app = Flask(__name__)
model = tf.keras.models.load_model('../model/model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_text = decode_prediction(prediction)
    return jsonify({'text': predicted_text})

if __name__ == '__main__':
    app.run(debug=True)

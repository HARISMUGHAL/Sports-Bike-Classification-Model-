from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

model = tf.keras.models.load_model('cnn_project2.h5')
app = Flask(__name__)

# Load class labels
dataset_path = 'project'
class_labels = sorted(os.listdir(dataset_path))

# Define a prediction function
def predict_image(image):
    # Preprocess the image
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)[0]
    confidence = np.max(predictions)
    predicted_class = class_labels[np.argmax(predictions)]

    # Check confidence and adjust predicted class if necessary
    if confidence < 0.6:  # 60% confidence threshold
        predicted_class = "NOT FROM SPORTS BIKES"

    return predicted_class, confidence

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        predicted_class, confidence = predict_image(image)
        return jsonify({
            'class': predicted_class,
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '_main_':
    app.run(debug=True)
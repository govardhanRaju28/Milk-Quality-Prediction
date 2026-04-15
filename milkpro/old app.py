from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)

# Load your pre-trained model
clf = joblib.load("C:/Users/Admin/Desktop/milkpro/model/random_forest_model2.pkl")  # Make sure to save your model first

def extract_color_features(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean_r = np.mean(image_rgb[:, :, 0])
    mean_g = np.mean(image_rgb[:, :, 1])
    mean_b = np.mean(image_rgb[:, :, 2])
    return [mean_r, mean_g, mean_b]

@app.route('/api/predict_milk_quality', methods=['POST'])
def predict_milk_quality():
    if 'image' not in request.files:
        print("request received:")
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if image is not None:
        print("solving")
        features = extract_color_features(image)
        prediction = clf.predict([features])[0]  # Predict "Good" or "Bad"
        return jsonify({'result': prediction})
    else:
        print("unabel to process")
        return jsonify({'error': 'Unable to process image'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

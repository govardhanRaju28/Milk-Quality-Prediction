from flask import Flask, request, jsonify
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Assuming the model is already trained
clf = RandomForestClassifier(n_estimators=100, random_state=42)
# If you saved the model, you can load it like this:
# clf = joblib.load("path_to_saved_model.pkl")

app = Flask(__name__)

def extract_color_features(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean_r = np.mean(image_rgb[:, :, 0])
    mean_g = np.mean(image_rgb[:, :, 1])
    mean_b = np.mean(image_rgb[:, :, 2])
    return [mean_r, mean_g, mean_b]

@app.route('/api/predict_milk_quality', methods=['POST'])
def predict_milk_quality():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if image is not None:
        features = extract_color_features(image)
        prediction = clf.predict([features])[0]
        return jsonify({'result': prediction})
    else:
        return jsonify({'error': 'Unable to process image'}), 500

if __name__ == '__main__':
    app.run(debug=True)

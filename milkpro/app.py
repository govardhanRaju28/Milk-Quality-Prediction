# Flask API to serve the ML model
from flask import Flask, request, jsonify
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops

# Load the trained model
model = joblib.load("C:/Users/Admin/Desktop/milkpro/model/random_forest_model2.pkl")

app = Flask(__name__)

def extract_color_features(image):
          
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mean_r = np.mean(image_rgb[:, :, 0])
            mean_g = np.mean(image_rgb[:, :, 1])
            mean_b = np.mean(image_rgb[:, :, 2])
            return [mean_r, mean_g, mean_b]
          

def extract_texture_features(image):
      
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            glcm = graycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            return [contrast, homogeneity, energy]
       

def extract_turbidity_features(image):
       
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            return [laplacian_var]
       

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image_file = request.files['image']
    image_np = np.fromfile(image_file, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    
    # Extract features from the image
    color_features = extract_color_features(image)
    texture_features = extract_texture_features(image)
    turbidity_features = extract_turbidity_features(image)
    
    # Combine features and make prediction
    features = color_features + texture_features + turbidity_features
    prediction = model.predict([features])
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL = load_model('model.h5')

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    img = np.expand_dims(img, axis=-1)
    return np.expand_dims(img, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    img = preprocess(filepath)
    predictions = MODEL.predict(img)
    class_index = np.argmax(predictions)
    return f'Predicted Class: {class_index}'

if __name__ == '__main__':
    app.run(debug=True)

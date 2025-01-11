import cv2
import numpy as np
from keras.models import load_model

# Load Model
model = load_model('model.h5')

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    return np.expand_dims(img, axis=-1)

# Start Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = cv2.resize(frame, (32, 32))
    img = preprocess(img)
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_index = np.argmax(predictions)

    cv2.putText(frame, f'Class: {class_index}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Traffic Sign Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

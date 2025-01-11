import os
import numpy as np
import cv2
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Constants
DATASET_PATH = 'Dataset'
LABELS_FILE = 'labels.csv'
IMAGE_SIZE = (32, 32, 3)
BATCH_SIZE = 32
EPOCHS = 10
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2

# Load Dataset
images, labels = [], []
label_map = pd.read_csv(LABELS_FILE, header=None).set_index(0)[1].to_dict()

for label, sign_name in label_map.items():
    folder_path = os.path.join(DATASET_PATH, str(label))
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
        images.append(img)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=TEST_RATIO)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_RATIO)

# Preprocess Data
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

X_train = np.array([preprocess(img) for img in X_train])
X_val = np.array([preprocess(img) for img in X_val])
X_test = np.array([preprocess(img) for img in X_test])

y_train = to_categorical(y_train, num_classes=len(label_map))
y_val = to_categorical(y_val, num_classes=len(label_map))
y_test = to_categorical(y_test, num_classes=len(label_map))

# Model Definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE
)

# Evaluate Model
loss, acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {acc}')

# Save Model
model.save('model.h5')

# Plot Training History
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

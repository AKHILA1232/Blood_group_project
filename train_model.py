import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

DATASET_PATH = "datasets"

X = []
y = []

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ridge_density = cv2.countNonZero(thresh)
    return [ridge_density]

for label in os.listdir(DATASET_PATH):
    folder = os.path.join(DATASET_PATH, label)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        X.append(extract_features(path))
        y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel="linear")
model.fit(X_train, y_train)

joblib.dump(model, "blood_group_model.pkl")

print("✅ Model trained and saved")

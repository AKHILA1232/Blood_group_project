import cv2
import joblib

model = joblib.load("blood_group_model.pkl")

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ridge_density = cv2.countNonZero(thresh)
    return [[ridge_density]]

# give test fingerprint image
image_path = "datasets/A/101_1.tif"

features = extract_features(image_path)
prediction = model.predict(features)

print("🩸 Predicted Blood Group:", prediction[0])

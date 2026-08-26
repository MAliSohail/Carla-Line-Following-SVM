from pathlib import Path
import cv2
import numpy as np
import joblib


IMAGE_PATH = Path("extracted_images/frame_020000.jpg")
MODEL_PATH = Path("models/svm_line_follower.pkl")

RESIZE_WIDTH = 80
RESIZE_HEIGHT = 40
ROI_START_RATIO = 0.45


def preprocess_image(image_path):
    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))

    h = img.shape[0]
    roi = img[int(h * ROI_START_RATIO):, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    features = edges.flatten().astype(np.float32) / 255.0
    return features.reshape(1, -1)


def label_to_text(label):
    if label == -1:
        return "LEFT"
    elif label == 1:
        return "RIGHT"
    else:
        return "STRAIGHT"


def main():
    model = joblib.load(MODEL_PATH)

    features = preprocess_image(IMAGE_PATH)
    prediction = model.predict(features)[0]

    print("Image:", IMAGE_PATH)
    print("Predicted label:", prediction)
    print("Direction:", label_to_text(prediction))


if __name__ == "__main__":
    main()
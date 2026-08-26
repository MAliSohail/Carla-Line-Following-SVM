from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


IMAGE_DIR = Path("extracted_images")
LABELS_CSV = Path("data/training_labels.csv")
MODEL_PATH = Path("models/svm_line_follower.pkl")
OUTPUT_CSV = Path("data/svm_batch_predictions.csv")

RESIZE_WIDTH = 80
RESIZE_HEIGHT = 40
ROI_START_RATIO = 0.45

# Test on a bigger random set
N_SAMPLES = 5000


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
    return features


def label_to_text(label):
    if label == -1:
        return "left"
    elif label == 1:
        return "right"
    return "straight"


def main():
    labels_df = pd.read_csv(LABELS_CSV)

    if len(labels_df) > N_SAMPLES:
        labels_df = labels_df.sample(n=N_SAMPLES, random_state=123)

    model = joblib.load(MODEL_PATH)

    rows = []
    X = []
    y_true = []

    skipped = 0

    print("Loading test images...")

    for _, row in labels_df.iterrows():
        image_path = IMAGE_DIR / row["frame"]

        if not image_path.exists():
            skipped += 1
            continue

        try:
            features = preprocess_image(image_path)
        except Exception:
            skipped += 1
            continue

        X.append(features)
        y_true.append(int(row["label"]))

        rows.append({
            "frame": row["frame"],
            "steering": row["steering"],
            "true_label": int(row["label"])
        })

    X = np.array(X)
    y_true = np.array(y_true)

    print(f"Loaded test samples: {len(X)}")
    print(f"Skipped: {skipped}")

    y_pred = model.predict(X)

    print("\nAccuracy:")
    print(accuracy_score(y_true, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[-1, 0, 1]))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=[-1, 0, 1]))

    results = pd.DataFrame(rows)
    results["predicted_label"] = y_pred
    results["true_direction"] = results["true_label"].apply(label_to_text)
    results["predicted_direction"] = results["predicted_label"].apply(label_to_text)
    results["correct"] = results["true_label"] == results["predicted_label"]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved predictions to: {OUTPUT_CSV}")

    print("\nWrong prediction examples:")
    print(results[results["correct"] == False].head(20))


if __name__ == "__main__":
    main()
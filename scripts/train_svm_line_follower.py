from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample


# ==========================
# SETTINGS
# ==========================

IMAGE_DIR = Path("extracted_images")
LABELS_CSV = Path("data/training_labels.csv")
MODEL_OUTPUT = Path("models/svm_line_follower.pkl")

# Image preprocessing settings
RESIZE_WIDTH = 80
RESIZE_HEIGHT = 40

# Use lower part of image because the road/line is usually there
ROI_START_RATIO = 0.45

# Set to True to balance left/straight/right classes
BALANCE_DATASET = True


def preprocess_image(image_path):
    """
    Convert an image into a feature vector for SVM.
    This must be reused later during live prediction.
    """

    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Resize
    img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT))

    # Crop region of interest
    h = img.shape[0]
    roi = img[int(h * ROI_START_RATIO):, :]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Smooth noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blur, 50, 150)

    # Normalize and flatten
    features = edges.flatten().astype(np.float32) / 255.0

    return features


def balance_dataset(X, y):
    """
    Downsample all classes to the size of the smallest class.
    This prevents the SVM from mostly predicting the majority class.
    """

    df = pd.DataFrame(X)
    df["label"] = y

    class_counts = df["label"].value_counts().sort_index()
    print("\nClass distribution before balancing:")
    print(class_counts)

    min_count = class_counts.min()

    balanced_parts = []

    for label in class_counts.index:
        class_df = df[df["label"] == label]

        class_downsampled = resample(
            class_df,
            replace=False,
            n_samples=min_count,
            random_state=42
        )

        balanced_parts.append(class_downsampled)

    balanced_df = pd.concat(balanced_parts).sample(frac=1, random_state=42)

    y_balanced = balanced_df["label"].values
    X_balanced = balanced_df.drop(columns=["label"]).values

    print("\nClass distribution after balancing:")
    print(pd.Series(y_balanced).value_counts().sort_index())

    return X_balanced, y_balanced


def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Labels CSV not found: {LABELS_CSV}")

    labels_df = pd.read_csv(LABELS_CSV)
    labels_df = labels_df.sample(n=9000, random_state=42)

    required_columns = {"frame", "steering", "label"}
    missing = required_columns - set(labels_df.columns)

    if missing:
        raise ValueError(f"Missing required columns in labels CSV: {missing}")

    X = []
    y = []

    skipped = 0

    print("Loading images and creating feature vectors...")

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
        y.append(int(row["label"]))

    X = np.array(X)
    y = np.array(y)

    print(f"\nLoaded samples: {len(X)}")
    print(f"Skipped samples: {skipped}")
    print(f"Feature vector size: {X.shape[1]}")

    print("\nOriginal class distribution:")
    print(pd.Series(y).value_counts().sort_index())

    if BALANCE_DATASET:
        X, y = balance_dataset(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining SVM...")

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            verbose=True
        )
    )

    model.fit(X_train, y_train)

    print("\nEvaluating model...")

    y_pred = model.predict(X_test)

    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)

    print(f"\nSaved trained model to: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
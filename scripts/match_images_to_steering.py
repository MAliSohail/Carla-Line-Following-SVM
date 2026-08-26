from pathlib import Path
import pandas as pd
import numpy as np


IMAGE_TIMESTAMPS_CSV = Path("data/image_timestamps.csv")
STEERING_RAW_CSV = Path("data/steering_raw.csv")
OUTPUT_CSV = Path("data/training_labels.csv")


def steering_to_label(steering, threshold=0.05):
    """
    Convert continuous steering values into 3 classes:
    -1 = left
     0 = straight
     1 = right
    """
    if steering < -threshold:
        return -1
    elif steering > threshold:
        return 1
    else:
        return 0


def main():
    image_df = pd.read_csv(IMAGE_TIMESTAMPS_CSV)
    steering_df = pd.read_csv(STEERING_RAW_CSV)

    # Make sure timestamps are sorted
    image_df = image_df.sort_values("timestamp").reset_index(drop=True)
    steering_df = steering_df.sort_values("timestamp").reset_index(drop=True)

    # Match each image to closest steering timestamp
    matched = pd.merge_asof(
        image_df,
        steering_df,
        on="timestamp",
        direction="nearest"
    )

    # Add class label
    matched["label"] = matched["steering"].apply(steering_to_label)

    # Optional: calculate quick statistics
    print("\nFirst rows:")
    print(matched.head())

    print("\nSteering summary:")
    print(matched["steering"].describe())

    print("\nClass distribution:")
    print(matched["label"].value_counts().sort_index())

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved matched training labels to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
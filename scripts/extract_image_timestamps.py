from pathlib import Path
import sys
import pandas as pd

from rosbags.rosbag2 import Reader


# ==========================
# CHANGE THESE SETTINGS
# ==========================

BAG_DIR = Path("enter_bag_dir")

CAMERA_TOPIC = "/carla/hero/camera/image/compressed"

IMAGE_DIR = Path("extracted_images")

OUTPUT_CSV = Path("data/image_timestamps.csv")


def main():
    if not BAG_DIR.exists():
        sys.exit(f"ERROR: Bag folder does not exist: {BAG_DIR}")

    if not IMAGE_DIR.exists():
        sys.exit(f"ERROR: Image folder does not exist: {IMAGE_DIR}")

    # Collect image names in the same sorted order as extraction
    image_files = sorted(
        list(IMAGE_DIR.glob("*.png")) +
        list(IMAGE_DIR.glob("*.jpg")) +
        list(IMAGE_DIR.glob("*.jpeg"))
    )

    if not image_files:
        sys.exit(f"ERROR: No images found in {IMAGE_DIR}")

    rows = []

    with Reader(BAG_DIR) as reader:
        connections = [
            c for c in reader.connections
            if c.topic == CAMERA_TOPIC
        ]

        if not connections:
            print(f"ERROR: Camera topic not found: {CAMERA_TOPIC}")
            print("\nAvailable topics:")
            for c in reader.connections:
                print(f"  {c.topic} | {c.msgtype}")
            return

        camera_timestamps = []

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            camera_timestamps.append(timestamp)

    print(f"Found {len(camera_timestamps)} camera messages in bag.")
    print(f"Found {len(image_files)} extracted images.")

    min_len = min(len(camera_timestamps), len(image_files))

    if len(camera_timestamps) != len(image_files):
        print("\nWARNING:")
        print("Number of camera messages and extracted images is not identical.")
        print("Using the smaller count to avoid mismatch.")
        print("This is okay if only a few frames differ, but check it carefully.")

    for image_path, timestamp in zip(image_files[:min_len], camera_timestamps[:min_len]):
        rows.append({
            "frame": image_path.name,
            "timestamp": timestamp
        })

    df = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved image timestamps to: {OUTPUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()

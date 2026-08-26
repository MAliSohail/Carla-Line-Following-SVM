from pathlib import Path
import sys
import cv2
import numpy as np

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


# ==========================
# CHANGE THESE SETTINGS
# ==========================

BAG_DIR = Path("test_session/bag")  
CAMERA_TOPIC = "/carla/hero/camera/image/compressed"

OUTPUT_DIR = Path("test_session/extracted_images")
IMAGE_EXTENSION = ".jpg"


def decode_compressed_image(msg):
    """
    Decode sensor_msgs/msg/CompressedImage into an OpenCV image.
    """

    np_arr = np.frombuffer(msg.data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode compressed image.")

    return image


def main():
    if not BAG_DIR.exists():
        sys.exit(f"ERROR: Bag folder does not exist: {BAG_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    frame_count = 0

    with Reader(BAG_DIR) as reader:
        print("\nAvailable topics:")
        for c in reader.connections:
            print(f"  {c.topic} | {c.msgtype}")

        connections = [
            c for c in reader.connections
            if c.topic == CAMERA_TOPIC
        ]

        if not connections:
            print(f"\nERROR: Camera topic not found: {CAMERA_TOPIC}")
            print("Change CAMERA_TOPIC to one of the image topics listed above.")
            return

        print(f"\nExtracting images from topic: {CAMERA_TOPIC}")
        print(f"Saving to: {OUTPUT_DIR}")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

            image = decode_compressed_image(msg)

            frame_count += 1
            filename = f"frame_{frame_count:06d}{IMAGE_EXTENSION}"
            output_path = OUTPUT_DIR / filename

            cv2.imwrite(str(output_path), image)

            if frame_count % 500 == 0:
                print(f"Extracted {frame_count} frames...")

    print(f"\nDone. Extracted {frame_count} frames.")
    print(f"Images saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
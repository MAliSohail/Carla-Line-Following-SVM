from pathlib import Path
import sys
import pandas as pd

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


# ==========================
# CHANGE THESE TWO SETTINGS
# ==========================

BAG_DIR = Path("enter_bag_dir")

STEERING_TOPIC = "/carla/hero/control_cmd"

OUTPUT_CSV = Path("data/steering_raw.csv")


def extract_steering(msg):
    """
    Try to extract steering from common CARLA / ROS control message formats.
    """

    # Common CARLA control message:
    # carla_msgs/msg/CarlaEgoVehicleControl
    if hasattr(msg, "steer"):
        return float(msg.steer)

    # Ackermann message:
    # ackermann_msgs/msg/AckermannDriveStamped
    if hasattr(msg, "drive") and hasattr(msg.drive, "steering_angle"):
        return float(msg.drive.steering_angle)

    # Geometry Twist fallback:
    # geometry_msgs/msg/Twist
    if hasattr(msg, "angular") and hasattr(msg.angular, "z"):
        return float(msg.angular.z)

    raise ValueError(
        "Could not find a steering field in this message. "
        "Expected something like msg.steer, msg.drive.steering_angle, or msg.angular.z."
    )


def main():
    if not BAG_DIR.exists():
        sys.exit(f"ERROR: Bag folder does not exist: {BAG_DIR}")

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    rows = []

    with Reader(BAG_DIR) as reader:
        available_topics = sorted(set(c.topic for c in reader.connections))

        print("\nAvailable topics in bag:")
        for topic in available_topics:
            print(" ", topic)

        connections = [
            c for c in reader.connections
            if c.topic == STEERING_TOPIC
        ]

        if not connections:
            print(f"\nERROR: Steering topic not found: {STEERING_TOPIC}")
            print("\nChoose the correct topic from the list above.")
            return

        print(f"\nReading steering topic: {STEERING_TOPIC}")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            try:
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                steering = extract_steering(msg)

                rows.append({
                    "timestamp": timestamp,
                    "steering": steering
                })

            except Exception as e:
                print("\nERROR while reading steering message:")
                print(e)
                print("\nMessage type was:")
                print(connection.msgtype)
                return

    if not rows:
        print("No steering values found.")
        return

    df = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(df)} steering values to:")
    print(OUTPUT_CSV)

    print("\nFirst few rows:")
    print(df.head())

    print("\nSteering value summary:")
    print(df["steering"].describe())


if __name__ == "__main__":
    main()

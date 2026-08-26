from pathlib import Path
import cv2


FRAMES_DIR = Path("visualization_output_opencv/frames")
OUTPUT_VIDEO = Path("visualization_output_opencv/opencv_replay.mp4")
FPS = 60


def main():
    frame_files = sorted(
        list(FRAMES_DIR.glob("*.jpg")) +
        list(FRAMES_DIR.glob("*.png"))
    )

    if not frame_files:
        raise FileNotFoundError(f"No frames found in: {FRAMES_DIR}")

    first_frame = cv2.imread(str(frame_files[0]))

    if first_frame is None:
        raise ValueError(f"Could not read first frame: {frame_files[0]}")

    height, width = first_frame.shape[:2]

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, FPS, (width, height))

    written = 0

    for frame_path in frame_files:
        frame = cv2.imread(str(frame_path))

        if frame is None:
            print(f"Skipping unreadable frame: {frame_path}")
            continue

        frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        written += 1

        if written % 1000 == 0:
            print(f"Written {written}/{len(frame_files)} frames...")

    writer.release()

    print("\nDone.")
    print(f"Frames written: {written}")
    print(f"Saved video to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
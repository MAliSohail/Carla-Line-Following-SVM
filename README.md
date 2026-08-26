# CARLA Lane-Following SVM

Predicting steering commands from camera frames using a support vector
machine, trained on driving data recorded in the CARLA simulator.

## Context

This was a university coursework project. The CARLA simulation environment, the ROS 2
integration and the recorded driving session were provided by the course.
This repository contains my own data pipeline and model: bag extraction,
timestamp alignment, labelling, class balancing, training and evaluation.

## Approach

Rather than deriving a steering target geometrically from where the line
appears in the frame, this pipeline learns from what the driver actually
did. The label for each camera frame is the steering command that was
issued at that moment, read directly from the recorded bag.

### Pipeline

1. **Extract steering** (`scripts/extract_steering_from_bag.py`)
   Reads `/carla/hero/control_cmd` from the ROS 2 bag using `rosbags`, so
   no ROS installation is required. Handles three possible message
   layouts (CARLA ego control, Ackermann, and Twist) by inspecting the
   fields present.

2. **Extract frame timestamps** (`scripts/extract_image_timestamps.py`)

3. **Align the two streams** (`scripts/match_images_to_steering.py`)
   Camera frames and control messages are published at different rates,
   so each frame is matched to its nearest steering value in time using
   `pandas.merge_asof`. Continuous steering is then binned into three
   classes with a deadband of 0.05: left, straight, right.

4. **Train** (`scripts/train_svm_line_follower.py`)
   Frames are resized to 80x40, cropped to the lower 55% where the road
   is, converted to greyscale, blurred, and passed through a Canny edge
   detector. The edge map is flattened to a 1760-dimensional vector.
   Training uses `StandardScaler` with an RBF-kernel `SVC` (C=10).

## Class imbalance

The recorded session was heavily skewed toward left and straight, since
the track curves predominantly one way. Trained directly on that
distribution, the model mostly predicts the majority class and scores
well while being useless.

All three classes are therefore downsampled to the size of the smallest
before training. This throws data away, which is the cost of the
approach; the alternative is mirroring frames to synthesise the
under-represented direction, which keeps the data but assumes the
left-right behaviour is symmetric.

## Results

Accuracy on a held-out 20% test split: **approximately 73%**, on three
balanced classes, where chance is 33%.

Training used a random sample of 9,000 labelled frames. An RBF-kernel SVM
scales roughly quadratically with the number of samples, so the full
dataset was not practical to fit on a laptop.

## Limitations

- The model is trained on a single driving session on one track, so it
  has not been tested for generalisation to a different route.
- Three-class output is coarse for smooth steering. Continuous
  regression would be the next thing to try.
- The deadband threshold of 0.05 was chosen by inspecting the steering
  distribution, not tuned systematically.

## Running it

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

The extraction scripts need the original bag, which is not distributed
here. The intermediate CSVs in `data/` are included, so the alignment and
labelling steps can be inspected without it.

The trained model is at `models/svm_line_follower.pkl`.

#!/usr/bin/env python3
"""Offline video: annotate frames with awake/sleepy and save crops + output video."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from drowsiness.config import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    FRAME_SKIP,
    OUTPUT_CROPS,
    OUTPUT_VIDEOS,
    SAMPLE_VIDEO_PATH,
    SLEEPY_THRESHOLD,
    SLEEPY_TIME_SEC,
)
from drowsiness.eye_crop import extract_eyes_from_frame  # noqa: E402
from drowsiness.preprocess import preprocess_eye  # noqa: E402


def main() -> None:
    model_path = DEFAULT_MODEL_PATH
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train via notebooks/02_baseline_cnn.ipynb or copy weights to models/."
        )

    model = load_model(model_path)
    print("Model loaded:", model_path)

    video_path = SAMPLE_VIDEO_PATH
    if not video_path.is_file():
        raise FileNotFoundError(
            f"Place a sample video at {video_path} (or edit SAMPLE_VIDEO_PATH in src/drowsiness/config.py)."
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    crop_dir = OUTPUT_CROPS / timestamp
    OUTPUT_VIDEOS.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = OUTPUT_VIDEOS / f"{timestamp}.mp4"
    out = cv2.VideoWriter(
        str(out_path),
        fourcc,
        fps,
        (int(cap.get(3)), int(cap.get(4))),
    )

    sleepy_start: float | None = None
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        label = "NO FACE DETECTED"
        color = (255, 255, 0)

        if frame_id % FRAME_SKIP != 0:
            out.write(frame)
            continue

        left_eye, right_eye, left_box, right_box = extract_eyes_from_frame(frame)

        if left_eye is not None and right_eye is not None:
            cv2.imwrite(str(crop_dir / f"left_{frame_id}.png"), left_eye)
            cv2.imwrite(str(crop_dir / f"right_{frame_id}.png"), right_eye)

            lp = float(model.predict(preprocess_eye(left_eye), verbose=0)[0][0])
            rp = float(model.predict(preprocess_eye(right_eye), verbose=0)[0][0])
            prob = (lp + rp) / 2

            if prob > SLEEPY_THRESHOLD:
                if sleepy_start is None:
                    sleepy_start = time.time()
                if time.time() - sleepy_start >= SLEEPY_TIME_SEC:
                    label, color = "SLEEPY", (0, 0, 255)
                else:
                    label, color = "CLOSING EYES", (0, 255, 255)
            else:
                sleepy_start = None
                label, color = "AWAKE", (0, 255, 0)

            lx, ly, lw, lh = left_box
            rx, ry, rw, rh = right_box
            cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), color, 2)
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)

        cv2.putText(
            frame,
            f"Status: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
        out.write(frame)

    cap.release()
    out.release()
    print(f"Done. Video: {out_path}")
    print(f"Crops: {crop_dir}")


if __name__ == "__main__":
    main()

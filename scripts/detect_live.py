#!/usr/bin/env python3
"""Webcam: live awake/sleepy overlay with eye boxes."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
from tensorflow.keras.models import load_model

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from drowsiness.config import (  # noqa: E402
    DEFAULT_MODEL_PATH,
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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (check permissions).")

    sleepy_start: float | None = None
    print("Live camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label = "No face"
        color = (255, 255, 0)

        left_eye, right_eye, left_box, right_box = extract_eyes_from_frame(frame)

        if left_eye is not None and right_eye is not None:
            lp = float(model.predict(preprocess_eye(left_eye), verbose=0)[0][0])
            rp = float(model.predict(preprocess_eye(right_eye), verbose=0)[0][0])
            prob = (lp + rp) / 2.0

            if prob > SLEEPY_THRESHOLD:
                if sleepy_start is None:
                    sleepy_start = time.time()
                elapsed = time.time() - sleepy_start
                if elapsed >= SLEEPY_TIME_SEC:
                    label, color = "SLEEPY", (0, 0, 255)
                else:
                    label, color = "Closing eyes", (0, 255, 255)
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
        cv2.imshow("Drowsiness detection (live)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera stopped.")


if __name__ == "__main__":
    main()
